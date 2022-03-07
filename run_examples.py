from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import logging
import os
import re
import shutil
import subprocess
import textwrap
import time

BOHR_TO_ANG = 0.529177

Atom = namedtuple("Atom", ["symbol", "charge", "x", "y", "z"])

class Wrapper:
    EXE = "/home/artfin/Desktop/QC-2022/firefly/firefly820"
    LIB = "/home/artfin/Desktop/QC-2022/firefly-examples/lib/"

    def __init__(self, wd=None, inpfname=None):
        self.wd = os.path.abspath(wd)

        if inpfname is None:
            inpfnames = [os.path.join(self.wd, f) for f in os.listdir(self.wd) if f.endswith(".fly")]
            assert len(inpfnames) == 1, "Detected not one inputfile in the working directory"
            self.inpfname = os.path.abspath(inpfnames[0])
        else:
            self.inpfname = os.path.join(self.wd, inpfname)
            self.basename = os.path.splitext(self.inpfname)[0]

        self.outfname = self.basename + '.out'

    def locate_group(self, lines, group):
        group_start = None 
        for n, line in enumerate(lines):
            if group in line:
                group_start = n
                break

        if group_start is None:
            return None, None

        group_end = None
        for n in range(group_start, len(lines)):
            line = lines[n]
            if '$END' in line:
                group_end = n
                break

        return group_start, group_end

    def load_inpfile(self):
        with open(self.inpfname) as inp:
            self.inp_code = inp.readlines()

    def set_options(self, options):
        for name, block_options in options.items():
            group_name = "$" + name.upper()
            self.set_block(group_name=group_name, options=block_options)

    @staticmethod
    def wrap_field(s, WIDTH=70):
        return "\n".join(textwrap.wrap(s, width=WIDTH, initial_indent=''))

    def set_block(self, group_name, options):
        group_start, group_end = self.locate_group(self.inp_code, group=group_name)
        if group_start is not None and group_end is not None:
            del self.inp_code[group_start : group_end + 1]
        else:
            group_start, _ = self.locate_group(self.inp_code, group="$DATA")

        if group_name == "$DATA":
            s = """ $DATA\n {}\n {}\n""".format(options["COMMENT"], options["SYMMETRY"])
            for atom in options["GEOMETRY"]:
                s += f"   {atom.symbol} {atom.charge:.1f} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}\n"

            s += " $END\n"
        else:
            s = " " + group_name
            for k, v in options.items():
                s += " {}={}".format(k, v)
            s += " $END"
            s = self.wrap_field(s) + "\n"

        self.inp_code.insert(group_start, s)


    def save_inpfile(self, fname):
        path = os.path.join(self.wd, fname)
        with open(path, mode='w') as out:
            for line in self.inp_code:
                out.write(line)

        self.inpfname = path
        self.basename = os.path.splitext(self.inpfname)[0]
        self.outfname = self.basename + '.out'

    def clean_up(self):
        tmp_dir = self.wd + '.0'
        assert os.path.isdir(tmp_dir)
        shutil.rmtree(tmp_dir)

        dat = os.path.splitext(self.inpfname)[0] + '.dat'
        assert os.path.isfile(dat)
        os.remove(dat)

    def clean_wd(self):
        ignore = lambda f: not (f.endswith(".fly") or f.endswith(".swp"))
        files = [os.path.join(self.wd, f) for f in os.listdir(self.wd) if ignore(f)]

        logging.info("Removing:")
        for f in files:
            if not f.startswith(self.basename):
                continue

            logging.info("  {}".format(f))
            os.remove(f)

    def run(self, link_basis=None):
        cmd = f"{self.EXE} -r -f -p -stdext -i {self.inpfname} -o {self.outfname} -t {self.wd}"

        if link_basis is not None:
            link_basis = link_basis.lower()
            basis_path = os.path.join(self.LIB, link_basis + '.lib')
            assert os.path.isfile(basis_path)
            cmd += f" -b {basis_path}"

        proc = subprocess.Popen(cmd, shell=True)
        while proc.poll() is None:
            time.sleep(0.5)

        assert proc.returncode == 0, "ERROR: Firefly program exited with non-zero exit code"


    def load_out(self):
        assert os.path.isfile(self.outfname)
        with open(self.outfname, mode='r') as inp:
            self.outfile = inp.readlines()

    def energy(self, method):
        lines = "".join(self.outfile)

        if method == "rhf":
            pattern = "TOTAL ENERGY = + ([-+]?\d+.\d+)"
            energy = re.findall(pattern, lines)
            assert len(energy) == 1
            energy = float(energy[0])
        elif method == "mp2":
            pattern = "E\(MP2\)= + ([-+]?\d+.\d+)"
            energy = re.findall(pattern, lines)
            assert len(energy) == 1
            energy = float(energy[0])

        else:
            raise ValueError("unreachable")

        return energy

    def frequencies(self):
        freqs = []
        for line in self.outfile:
            if "FREQUENCY" in line:
                words = line.split()[1:]
                freqs.extend(list(map(float, words)))

        return freqs

    def opt_geometries(self, natoms=3):
        geometries = []
        for ind, line in enumerate(self.outfile):
            if "COORDINATES OF ALL ATOMS ARE" in line:
                geom = []
                for k in range(natoms):
                    words = self.outfile[ind+k+3].split()
                    atom = Atom(symbol=words[0], charge=int(float(words[1])), x=float(words[2]), y=float(words[3]), z=float(words[4]))
                    geom.append(atom)

                geometries.append(geom)

        return geometries

    def thermo(self):
        lines = "".join(self.outfile)

        pattern = "THERMOCHEMISTRY AT T= +(\d+.\d+) K"
        temperatures = re.findall(pattern, lines)
        temperatures = list(map(float, temperatures))

        pattern = """E         H         G         CV        CP        S
       + KCAL/MOL  KCAL/MOL  KCAL/MOL CAL/MOL-K CAL/MOL-K CAL/MOL-K
 ELEC. + \d+.\d+ + \d+.\d+ + \d+.\d+ + \d+.\d+ + \d+.\d+ + \d+.\d+
 TRANS. + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+
 ROT. + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+
 VIB. + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+
 TOTAL. + ([-+]?\d+.\d+) + ([-+]?\d+.\d+) + ([-+]?\d+.\d+) + ([-+]?\d+.\d+) + ([-+]?\d+.\d+) + ([-+]?\d+.\d+)"""

        quantities = ("E", "H", "G", "CV", "CP", "S")
        total_matches = re.findall(pattern, lines)

        assert len(temperatures) == len(total_matches)

        total_thermo = {}
        for T, matches in zip(temperatures, total_matches):
            total_thermo[T] = {q : float(m) for q, m in zip(quantities, matches)}

        return total_thermo


def run_example_01():
    logging.info(" --- CO2 RHF ENERGY CALCULATION USING BASIS=STO-3G --- ")
    wrapper = Wrapper(wd="1_co2_en", inpfname="1_co2_en.fly")
    wrapper.load_inpfile()
    wrapper.set_options({
        "basis" : {"GBASIS": "STO", "NGAUSS": 3},
    })
    wrapper.save_inpfile("1_co2_rhf_en-basis=sto-3g.fly")

    wrapper.clean_wd()
    wrapper.run()
    wrapper.clean_up()

    wrapper.load_out()
    rhf_energy = wrapper.energy(method="rhf")
    logging.info("RHF Energy: {}".format(rhf_energy))
    logging.info("---------------------------------------------------------\n")

    #############################################################################

    logging.info(" --- CO2 RHF ENERGY CALCULATION USING BASIS=CC-PVDZ --- ")
    wrapper = Wrapper(wd="1_co2_en", inpfname="1_co2_en.fly")
    wrapper.load_inpfile()
    wrapper.set_options({
        "basis": {"GBASIS": "CC-PVDZ", "EXTFILE": ".T."},
    })
    wrapper.save_inpfile("1_co2_rhf_en-basis=cc-pvdz.fly")

    wrapper.clean_wd()
    wrapper.run(link_basis="cc-pvdz")
    wrapper.clean_up()

    wrapper.load_out()
    rhf_energy = wrapper.energy(method="rhf")
    logging.info("RHF Energy: {}".format(rhf_energy))
    logging.info("---------------------------------------------------------\n")

    #############################################################################

    logging.info(" --- CO2 MP2 ENERGY CALCULATION USING BASIS=CC-PVDZ --- ")
    wrapper = Wrapper(wd="1_co2_en", inpfname="1_co2_en.fly")
    wrapper.load_inpfile()
    wrapper.set_options({
        "contrl": {"SCFTYP": "RHF", "MPLEVL": 2, "MULT": 1, "UNITS": "BOHR"},
        "basis" : {"GBASIS": "CC-PVDZ", "EXTFILE": ".T."},
        "mp2"   : {"METHOD": 1},
        "scf"   : {"DIRSCF": ".T.", "DIIS": ".T.", "NCONV": 8, "ENGTHR": 9}
    })
    wrapper.save_inpfile("1_co2_mp2_en-basis=cc=pvdz.fly")

    wrapper.clean_wd()
    wrapper.run(link_basis="cc-pvdz")
    wrapper.clean_up()

    wrapper.load_out()
    rhf_energy = wrapper.energy(method="rhf")
    mp2_energy = wrapper.energy(method="mp2")
    logging.info("RHF Energy: {}".format(rhf_energy))
    logging.info("MP2 Energy: {}".format(mp2_energy))
    logging.info("---------------------------------------------------------\n")
    #############################################################################


def run_example_02():
    #logging.info(" --- CO2 RHF GEOMETRY OPTIMIZATION USING BASIS=STO-3G --- ")
    #wrapper = Wrapper(wd="2_co2_opt", inpfname="2_co2_opt.fly")

    #wrapper.load_inpfile()
    #wrapper.set_options({
    #    "contrl": {"SCFTYP": "RHF", "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR"},
    #    "basis" : {"GBASIS": "STO", "NGAUSS": 3},
    #})
    #wrapper.save_inpfile("2_co2_rhf_opt-basis=sto-3g.fly")

    #wrapper.clean_wd()
    #wrapper.run()
    #wrapper.clean_up()

    #wrapper.load_out()
    #geometries = wrapper.opt_geometries()

    #opt = geometries[-1]
    #logging.info("Optimized geometry (ANG):")
    #for atom in opt:
    #    logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")

    #opt = [Atom(symbol=atom.symbol, charge=atom.charge, x=atom.x/BOHR_TO_ANG, y=atom.y/BOHR_TO_ANG, z=atom.z/BOHR_TO_ANG)
    #       for atom in opt]

    #logging.info("Optimized geometry (BOHR):")
    #for atom in opt:
    #    logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")

    #############################################################################

    #logging.info(" --- CO2 RHF HESSIAN VERIFICATION USING BASIS=STO-3G --- ")
    #wrapper = Wrapper(wd="2_co2_opt", inpfname="2_co2_hess.fly")

    #wrapper.load_inpfile()
    #wrapper.set_options({
    #    "contrl" : {"SCFTYP": "RHF", "RUNTYP": "HESSIAN", "MULT": 1 , "UNITS": "BOHR"},
    #    "basis"  : {"GBASIS": "STO", "NGAUSS": 3},
    #    "data"   : {"COMMENT": "CO2 HESSIAN AT OPT", "SYMMETRY": "C1", "GEOMETRY": opt}
    #})
    #wrapper.save_inpfile("2_co2_rhf_hess-basis=sto-3g.fly")

    #wrapper.clean_wd()
    #wrapper.run()
    #wrapper.clean_up()

    #wrapper.load_out()
    #freqs = wrapper.frequencies()

    #logging.info("Frequencies at optimized geometry (cm-1):")
    #for f in freqs:
    #    logging.info("  {:.3f}".format(f))

    #positive = all(f > 0.0 for f in freqs)
    #logging.info("Assert freqs > 0: {}".format(positive))
    #assert positive

    #############################################################################

    logging.info(" --- CO2 RHF GEOMETRY OPTIMIZATION USING BASIS=CC-PVDZ --- ")
    wrapper = Wrapper(wd="2_co2_opt", inpfname="2_co2_opt.fly")

    wrapper.load_inpfile()
    wrapper.set_options({
        "contrl": {"SCFTYP": "RHF", "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR"},
        "basis" : {"GBASIS": "CC-PVDZ", "EXTFILE": ".T."},
    })
    wrapper.save_inpfile("2_co2_rhf_opt-basis=cc-pvdz.fly")

    wrapper.clean_wd()
    wrapper.run(link_basis='cc-pvdz')
    wrapper.clean_up()

    wrapper.load_out()
    geometries = wrapper.opt_geometries()

    opt = geometries[-1]
    logging.info("Optimized geometry (ANG):")
    for atom in opt:
        logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")

    opt = [Atom(symbol=atom.symbol, charge=atom.charge, x=atom.x/BOHR_TO_ANG, y=atom.y/BOHR_TO_ANG, z=atom.z/BOHR_TO_ANG)
           for atom in opt]

    logging.info("Optimized geometry (BOHR):")
    for atom in opt:
        logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")

    #############################################################################

    logging.info(" --- CO2 RHF HESSIAN VERIFICATION USING BASIS=CC-PVDZ --- ")
    wrapper = Wrapper(wd="2_co2_opt", inpfname="2_co2_hess.fly")

    wrapper.load_inpfile()
    wrapper.set_options({
        "contrl" : {"SCFTYP": "RHF", "RUNTYP": "HESSIAN", "MULT": 1 , "UNITS": "BOHR"},
        "basis"  : {"GBASIS": "CC-PVDZ", "EXTFILE": ".T."},
        "data"   : {"COMMENT": "CO2 HESSIAN AT OPT", "SYMMETRY": "C1", "GEOMETRY": opt}
    })
    wrapper.save_inpfile("2_co2_rhf_hess-basis=cc-pvdz.fly")

    wrapper.clean_wd()
    wrapper.run(link_basis='cc-pvdz')
    wrapper.clean_up()

    wrapper.load_out()
    freqs = wrapper.frequencies()

    logging.info("Frequencies at optimized geometry (cm-1):")
    for f in freqs:
        logging.info("  {:.3f}".format(f))

    positive = all(f > 0.0 for f in freqs)
    logging.info("Assert freqs > 0: {}".format(positive))
    assert positive
    #############################################################################

    logging.info(" --- CO2 MP2 GEOMETRY OPTIMIZATION USING BASIS=CC-PVDZ --- ")
    wrapper = Wrapper(wd="2_co2_opt", inpfname="2_co2_opt.fly")

    wrapper.load_inpfile()
    wrapper.set_options({
        "contrl": {"SCFTYP": "RHF", "MPLEVL": 2, "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR"},
        "basis" : {"GBASIS": "CC-PVDZ", "EXTFILE": ".T."},
        "mp2"   : {"METHOD": 1},
    })
    wrapper.save_inpfile("2_co2_mp2_opt-basis=cc-pvdz.fly")

    wrapper.clean_wd()
    wrapper.run(link_basis='cc-pvdz')
    wrapper.clean_up()

    wrapper.load_out()
    geometries = wrapper.opt_geometries()

    opt = geometries[-1]
    logging.info("Optimized geometry (ANG):")
    for atom in opt:
        logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")

    opt = [Atom(symbol=atom.symbol, charge=atom.charge, x=atom.x/BOHR_TO_ANG, y=atom.y/BOHR_TO_ANG, z=atom.z/BOHR_TO_ANG)
           for atom in opt]

    logging.info("Optimized geometry (BOHR):")
    for atom in opt:
        logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")
    #############################################################################

    logging.info(" --- CO2 MP2 HESSIAN VERIFICATION USING BASIS=CC-PVDZ --- ")
    wrapper = Wrapper(wd="2_co2_opt", inpfname="2_co2_hess.fly")

    wrapper.load_inpfile()
    wrapper.set_options({
        "contrl" : {"SCFTYP": "RHF", "MPLEVL": 2, "RUNTYP": "HESSIAN", "MULT": 1 , "UNITS": "BOHR"},
        "basis"  : {"GBASIS": "CC-PVDZ", "EXTFILE": ".T."},
        "mp2"    : {"METHOD": 1},
        "data"   : {"COMMENT": "CO2 HESSIAN AT OPT", "SYMMETRY": "C1", "GEOMETRY": opt}
    })
    wrapper.save_inpfile("2_co2_mp2_hess-basis=cc-pvdz.fly")

    wrapper.clean_wd()
    wrapper.run(link_basis='cc-pvdz')
    wrapper.clean_up()

    wrapper.load_out()
    freqs = wrapper.frequencies()

    logging.info("Frequencies at optimized geometry (cm-1):")
    for f in freqs:
        logging.info("  {:.3f}".format(f))

    positive = all(f > 0.0 for f in freqs)
    logging.info("Assert freqs > 0: {}".format(positive))
    assert positive
    #############################################################################

def NIST_CO2_CP(T):
    """
    Source:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=1&Type=JANAFG&Table=on#JANAFG
    Chase, M.W., Jr., NIST-JANAF Themochemical Tables, Fourth Edition,
    J. Phys. Chem. Ref. Data, Monograph 9, 1998, 1-1951.
    """
    A = 24.99735
    B = 55.18696
    C = -33.69137
    D = 7.948387
    E = -0.136638

    JTOKCAL = 0.2390057361
    t = T / 1000.0
    return JTOKCAL * (A + B * t + C * t**2 + D * t**3 + E * t**(-2)) # cal/mol*K


def run_example_03():
    gbasis = 'CC-PVDZ'

    logging.info(f" --- CO2 RHF THERMOCHEMISTRY USING BASIS={gbasis} --- ")
    wrapper = Wrapper(wd="3_co2_thermo", inpfname="3_co2_thermo.fly")

    wrapper.load_inpfile()

    temperatures = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    temperature_field = ", ".join(list(map(str, temperatures)))

    wrapper.set_options({
        "contrl": {"SCFTYP": "RHF", "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR",
                   "ICUT": 11, "INTTYP": "HONDO", "MAXIT": 100},
        "basis" : {"GBASIS": gbasis, "EXTFILE": ".T."},
        "statpt": {"METHOD": "GDIIS", "UPHESS": "BFGS", "OPTTOL": 1e-5, "HSSEND": ".T."},
        "scf"   : {"DIRSCF": ".T.", "DIIS": ".T.", "NCONV": 8, "ENGTHR": 9, "FDIFF": ".F."},
        "force" : {"TEMP(1)" : temperature_field}
    })
    wrapper.save_inpfile(f"3_co2_rhf_thermo-basis={gbasis}.fly")

    wrapper.clean_wd()
    wrapper.run(link_basis=gbasis)
    wrapper.clean_up()

    wrapper.load_out()
    thermo = wrapper.thermo()

    Cp_RHF = [block["CP"] for _, block in thermo.items()]
    print("Cp_RHF: {}".format(Cp_RHF))

    #############################################################################

    logging.info(f" --- CO2 MP2 THERMOCHEMISTRY USING BASIS={gbasis} --- ")
    wrapper = Wrapper(wd="3_co2_thermo", inpfname="3_co2_thermo.fly")

    wrapper.load_inpfile()

    temperatures = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    temperature_field = ", ".join(list(map(str, temperatures)))

    wrapper.set_options({
        "contrl": {"SCFTYP": "RHF", "MPLEVL": 2, "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR",
                   "ICUT": 11, "INTTYP": "HONDO", "MAXIT": 100},
        "basis" : {"GBASIS": gbasis, "EXTFILE": ".T."},
        "mp2"   : {"METHOD": 1},
        "statpt": {"METHOD": "GDIIS", "UPHESS": "BFGS", "OPTTOL": 1e-5, "HSSEND": ".T."},
        "scf"   : {"DIRSCF": ".T.", "DIIS": ".T.", "NCONV": 8, "ENGTHR": 9, "FDIFF": ".F."},
        "force" : {"TEMP(1)" : temperature_field}
    })
    wrapper.save_inpfile(f"3_co2_mp2_thermo-basis={gbasis}.fly")

    wrapper.clean_wd()
    wrapper.run(link_basis=gbasis)
    wrapper.clean_up()

    wrapper.load_out()
    thermo = wrapper.thermo()

    Cp_MP2 = [block["CP"] for _, block in thermo.items()]
    print("Cp_MP2: {}".format(Cp_MP2))

    #############################################################################

    Cp_NIST = np.asarray([NIST_CO2_CP(t) for t in temperatures])
    print("Cp_NIST:       {}".format(Cp_NIST))

    plt.figure(figsize=(10, 10))

    plt.plot(temperatures, Cp_RHF, color='y', label=f"HF/{gbasis}")
    plt.plot(temperatures, Cp_MP2, color='b', label=f"MP2/{gbasis}")
    plt.plot(temperatures, Cp_NIST, color='r', label="NIST")

    plt.legend(fontsize=14)
    plt.xlabel("Temperature, K")
    plt.ylabel("Heat capacity, kcal/mol")

    plt.show()
    logging.info("---------------------------------------------------------\n")

def run_example_04():
    gbasis = 'CC-PVDZ'

    #Source: https://cccbdb.nist.gov/vibscalejust.asp
    SCLFAC = {'HF/CC-PVDZ': 0.908, 'HF/CC-PVTZ': 0.9101, 'HF/CC-PVQZ': 0.9084,
              'MP2/CC-PVDZ': 0.953, 'MP2/CC-PVTZ': 0.950, 'MP2/CC-PVQZ': 0.948,}

    #############################################################################
    sclfac = SCLFAC[f'HF/{gbasis}']
    logging.info(f" --- CO2 RHF THERMOCHEMISTRY USING BASIS={gbasis} AND SCALING FACTOR={sclfac} --- ")
    wrapper = Wrapper(wd="4_co2_thermo_sclfac", inpfname="4_co2_thermo_sclfac.fly")

    wrapper.load_inpfile()

    temperatures = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    temperature_field = ", ".join(list(map(str, temperatures)))

    wrapper.set_options({
        "contrl": {"SCFTYP": "RHF", "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR",
                   "ICUT": 11, "INTTYP": "HONDO", "MAXIT": 100},
        "basis" : {"GBASIS": gbasis, "EXTFILE": ".T."},
        "statpt": {"METHOD": "GDIIS", "UPHESS": "BFGS", "OPTTOL": 1e-5, "HSSEND": ".T."},
        "scf"   : {"DIRSCF": ".T.", "DIIS": ".T.", "NCONV": 8, "ENGTHR": 9, "FDIFF": ".F."},
        "force" : {"TEMP(1)" : temperature_field, "SCLFAC": sclfac}
    })

    wrapper.save_inpfile(f"4_co2_rhf_thermo_sclfac-basis={gbasis}.fly")

    #wrapper.clean_wd()
    #wrapper.run(link_basis=gbasis)
    #wrapper.clean_up()

    wrapper.load_out()
    thermo = wrapper.thermo()

    Cp_RHF = [block["CP"] for _, block in thermo.items()]
    print("Cp_RHF: {}".format(Cp_RHF))

    #############################################################################
    sclfac = SCLFAC[f'MP2/{gbasis}']
    logging.info(f" --- CO2 MP2 THERMOCHEMISTRY USING BASIS={gbasis} AND SCALING FACTOR={sclfac} --- ")
    wrapper = Wrapper(wd="4_co2_thermo_sclfac", inpfname="4_co2_thermo_sclfac.fly")

    wrapper.load_inpfile()

    temperatures = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    temperature_field = ", ".join(list(map(str, temperatures)))
    sclfac = SCLFAC[f'MP2/{gbasis}']

    wrapper.set_options({
        "system": {"TIMLIM": 2880, "MEMORY": 8000000},
        "contrl": {"SCFTYP": "RHF", "MPLEVL": 2, "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR",
                   "ICUT": 11, "INTTYP": "HONDO", "MAXIT": 100},
        "basis" : {"GBASIS": gbasis, "EXTFILE": ".T."},
        "mp2"   : {"METHOD": 1},
        "statpt": {"METHOD": "GDIIS", "UPHESS": "BFGS", "OPTTOL": 1e-5, "HSSEND": ".T."},
        "scf"   : {"DIRSCF": ".T.", "DIIS": ".T.", "NCONV": 8, "ENGTHR": 9, "FDIFF": ".F."},
        "force" : {"TEMP(1)" : temperature_field, "SCLFAC": sclfac}
    })

    wrapper.save_inpfile(f"4_co2_mp2_thermo_sclfac-basis={gbasis}.fly")

    #wrapper.clean_wd()
    #wrapper.run(link_basis=gbasis)
    #wrapper.clean_up()

    wrapper.load_out()
    thermo = wrapper.thermo()

    Cp_MP2 = [block["CP"] for _, block in thermo.items()]
    print("Cp_MP2: {}".format(Cp_MP2))
    #############################################################################

    Cp_NIST = np.asarray([NIST_CO2_CP(t) for t in temperatures])
    print("Cp_NIST:       {}".format(Cp_NIST))

    plt.figure(figsize=(10, 10))

    plt.plot(temperatures, Cp_RHF, color='y', label=f"HF/{gbasis}+sclfac")
    plt.plot(temperatures, Cp_MP2, color='b', label=f"MP2/{gbasis}+sclfac")
    plt.plot(temperatures, Cp_NIST, color='r', label="NIST")

    plt.legend(fontsize=14)
    plt.xlabel("Temperature, K")
    plt.ylabel("Heat capacity, kcal/mol")

    figpath = os.path.join("4_co2_thermo_sclfac", f"CP-RHF-vs-MP2-basis={gbasis}.png")
    plt.savefig(figpath, format='png', dpi=300)

    plt.show()

    logging.info("---------------------------------------------------------\n")


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #run_example_01()
    #run_example_02()
    #run_example_03()
    run_example_04()



