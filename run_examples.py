from collections import namedtuple

import numpy as np

import logging
import os
import re
import shutil
import subprocess
import textwrap
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams["text.usetex"] =True
plt.rcParams["mathtext.fontset"] = "cm"

mpl.rcParams['font.serif'] = 'Times'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['axes.labelsize'] = 21
mpl.rcParams['axes.titlesize'] = 21
mpl.rcParams['xtick.labelsize'] = 21
mpl.rcParams['ytick.labelsize'] = 21

Boltzmann    = 1.380649e-23        # SI: J / K
Planck       = 6.62607015e-34      # SI: J * s
Dalton       = 1.6605300000013e-27 # SI: kg
UGC          = 8.31446261815324    # SI: m^3 * Pa / K / mol
a0           = 5.29177210903e-11   # SI: m
SpeedOfLight = 299792458.0         # SI: m/s

BOHR_TO_ANG = 0.529177
CAL_TO_J    = 4.184

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

    def trans_partf(self, T):
        lines = "".join(self.outfile)

        pattern ="""ATOMIC WEIGHTS \(AMU\)\n\n (?: +\d+ +\w +\d+.\d+\n)+"""
        m = re.findall(pattern, lines)[0]

        m = m.split('\n')[2:-1]
        masses = [float(line.split()[2]) for line in m]

        CM = sum(masses)

        p = 1.01325E+05 # pascal
        V = Boltzmann * T / p
        trans = (2.0 * np.pi * CM * Dalton * Boltzmann * T / Planck**2)**1.5 * V

        return trans

    def trans_cp(self, T):
        return 5.0 / 2.0 * UGC

    def rot_cp_cl(self, T):
        return UGC

    JMAX = 100
    def rot_partf_q(self, T):
        IT = 158.00455 * Dalton * a0**2
        theta_rot = Planck * Planck / (8.0 * np.pi**2 * IT * Boltzmann)

        qrot = 0.0
        for J in range(0, self.JMAX, 2): # due to nuclear statistics only even J are allowed
            qrot += (2 * J + 1) * np.exp(-J * (J + 1) * theta_rot / T)

        return qrot

    def rot_partf_cl(self, T):
        lines = "".join(self.outfile)

        pattern = """THE MOMENTS OF INERTIA ARE \(IN AMU\*BOHR\*\*2\)\n +(\d+.\d+) +(\d+.\d+) +(\d+.\d+)"""
        m = re.search(pattern, lines).groups()

        IT = list(map(float, m))
        assert(IT[1] == IT[2] and IT[0] == 0)

        ITT = IT[1]
        ITT = ITT * Dalton * a0**2

        Sigma = 2.0 # symmetry number
        qrot = 8.0 * np.pi**2 * ITT * Boltzmann * T / Sigma / Planck**2
        return qrot

    def rot_int_energy_cl(self, T):
        return UGC * T

    def rot_int_energy_q(self, T):
        dT = 1e-5
        #der = (np.log(self.rot_partf_q(T + dT)) - np.log(self.rot_partf_q(T))) / dT
        der = (np.log(self.rot_partf_q(T + dT)) - np.log(self.rot_partf_q(T - dT))) / (2.0 * dT)

        return UGC * T**2 * der


    def rot_cp_q(self, T):
        dT = 1e-5
        return (self.rot_int_energy_q(T + dT) - self.rot_int_energy_q(T - dT)) / (2.0 * dT)

    def rot_cp_EM1(self, T):
        IT = 158.00455 * Dalton * a0**2
        theta_rot = Planck * Planck / (8.0 * np.pi**2 * IT * Boltzmann)

        return 3.0 * UGC * T * (3.0 * T + 2.0 * theta_rot) / (3.0 * T + theta_rot)**2

    def rot_cp_EM2(self, T):
        IT = 158.00455 * Dalton * a0**2
        theta_rot = Planck * Planck / (8.0 * np.pi**2 * IT * Boltzmann)

        return UGC * (225.0 * T**4 + 150.0 * T**3 * theta_rot + 60.0 * T**2 * theta_rot**2 - theta_rot**4) / (15.0 * T**2 + 5.0 * theta_rot * T + theta_rot**2)**2

    def rot_cp_EM3(self, T):
        IT = 158.00455 * Dalton * a0**2
        theta_rot = Planck * Planck / (8.0 * np.pi**2 * IT * Boltzmann)

        return UGC * (99225.0 * T**6 + 66150.0 * T**5 * theta_rot + 26460.0 * T**4 * theta_rot**2 + 10080.0 * T**3 * theta_rot**3 + 399.0 * T**2 * theta_rot**4 - 168.0 * T * theta_rot**5 - 32.0 * theta_rot**6) / (315.0 * T**3 + 105.0 * theta_rot * T**2 + 21.0 * theta_rot**2 * T + 4.0 * theta_rot**3)**2

    def rot_cp_EM4(self, T):
        IT = 158.00455 * Dalton * a0**2
        theta_rot = Planck * Planck / (8.0 * np.pi**2 * IT * Boltzmann)

        return UGC * (99225.0 * T**8 + 66150.0 * T**7 * theta_rot + 26460.0 * T**6 * theta_rot**2 + 10080.0 * T**5 * theta_rot**3 + 4809.0 * T**4 * theta_rot**4 + 462.0 * T**3 * theta_rot**5 - 32.0 * T**2 * theta_rot**6 - 16.0 * T * theta_rot**7 - 3 * theta_rot**8) / (315.0 * T**4 + 105.0 * theta_rot * T**3 + 21.0 * theta_rot**2 * T**2 + 4.0 * theta_rot**3 * T + theta_rot**4)**2

    def vib_cp_q(self, T):
        freqs = self.frequencies()[5:]

        cp = 0.0
        for f in freqs:
            x = Planck * SpeedOfLight * 100.0 * f / Boltzmann / T
            cp += x**2 * np.exp(-x) / (np.exp(-x) - 1)**2

        return UGC * cp

    def vib_cp_cl(self, T):
        return 4.0 * UGC

    def vib_partf_q(self, T):
        freqs = self.frequencies()[5:]

        qvib = 1.0
        for f in freqs:
            a = Planck * SpeedOfLight * 100.0 * f / Boltzmann / T
            qvib *= 1.0 / (1.0 - np.exp(-a))

        return qvib

    def vib_partf_cl(self, T):
        freqs = self.frequencies()[5:]

        qvib = 1.0
        for f in freqs:
            qvib *= Boltzmann * T / (Planck * SpeedOfLight * 100.0 * f)

        return qvib

    def rot_entropy_q(self, T):
        return self.rot_int_energy_q(T) / T + UGC * self.rot_partf_q(T)

    def rot_entropy_cl(self, T):
        return UGC * np.log(self.rot_partf_cl(T)) + UGC

    def trans_entropy(self, T):
        lines = "".join(self.outfile)

        pattern ="""ATOMIC WEIGHTS \(AMU\)\n\n (?: +\d+ +\w +\d+.\d+\n)+"""
        m = re.findall(pattern, lines)[0]

        m = m.split('\n')[2:-1]
        masses = [float(line.split()[2]) for line in m]

        CM = sum(masses)

        # Source: https://mipt.ru/dbmp/upload/088/glava3-arphlf43020.pdf (p.49; eq. 3.16)
        p = 1.0 # atm
        return 1.5 * UGC * np.log(CM) + 2.5 * UGC * np.log(T) - UGC * np.log(p) - 9.7

    def vib_entropy_q(self, T):
        freqs = self.frequencies()[5:]

        svib = 0.0
        for f in freqs:
            a = Planck * SpeedOfLight * 100.0 * f / Boltzmann / T
            svib = svib + a * np.exp(-a) / (1 - np.exp(-a)) - np.log(1 - np.exp(-a))

        return UGC * svib


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

    #logging.info(" --- CO2 RHF GEOMETRY OPTIMIZATION USING BASIS=CC-PVDZ --- ")
    #wrapper = Wrapper(wd="2_co2_opt", inpfname="2_co2_opt.fly")

    #wrapper.load_inpfile()
    #wrapper.set_options({
    #    "contrl": {"SCFTYP": "RHF", "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR"},
    #    "basis" : {"GBASIS": "CC-PVDZ", "EXTFILE": ".T."},
    #    "statpt" : {"METHOD": "GDIIS", "UPHESS": "BFGS", "OPTTOL": 1e-5, "HSSEND": ".T."}
    #})
    #wrapper.save_inpfile("2_co2_rhf_opt-basis=cc-pvdz.fly")

    #wrapper.clean_wd()
    #wrapper.run(link_basis='cc-pvdz')
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

   # logging.info(" --- CO2 RHF HESSIAN VERIFICATION USING BASIS=CC-PVDZ --- ")
   # wrapper = Wrapper(wd="2_co2_opt", inpfname="2_co2_hess.fly")

   # wrapper.load_inpfile()
   # wrapper.set_options({
   #     "contrl" : {"SCFTYP": "RHF", "RUNTYP": "HESSIAN", "MULT": 1 , "UNITS": "BOHR"},
   #     "basis"  : {"GBASIS": "CC-PVDZ", "EXTFILE": ".T."},
   #     "data"   : {"COMMENT": "CO2 HESSIAN AT OPT", "SYMMETRY": "C1", "GEOMETRY": opt}
   # })
   # wrapper.save_inpfile("2_co2_rhf_hess-basis=cc-pvdz.fly")

   # wrapper.clean_wd()
   # wrapper.run(link_basis='cc-pvdz')
   # wrapper.clean_up()

   # wrapper.load_out()
   # freqs = wrapper.frequencies()

   # logging.info("Frequencies at optimized geometry (cm-1):")
   # for f in freqs:
   #     logging.info("  {:.3f}".format(f))

   # positive = all(f > 0.0 for f in freqs)
   # logging.info("Assert freqs > 0: {}".format(positive))
   # assert positive
    #############################################################################

    basis = "CC-PVQZ"
    logging.info(" --- CO2 MP2 GEOMETRY OPTIMIZATION USING BASIS={} --- ".format(basis))
    wrapper = Wrapper(wd="2_co2_opt", inpfname="2_co2_opt.fly")

    wrapper.load_inpfile()
    wrapper.set_options({
        "contrl" : {"SCFTYP": "RHF", "MAXIT": 100, "MPLEVL": 2, "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR",
                    "ICUT": 11, "INTTYP": "HONDO", "MAXIT": 100},
        "system" : {"memory" : 12000000},
        "basis"  : {"GBASIS": basis, "EXTFILE": ".T."},
        "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "NCONV": 8, "ENGTHR": 9, "FDIFF": ".F."},
        "mp2"    : {"METHOD": 1},
        "statpt" : {"METHOD": "GDIIS", "UPHESS": "BFGS", "OPTTOL": 1e-5},
    })
    wrapper.save_inpfile("2_co2_mp2_opt-basis={}.fly".format(basis.lower()))

    wrapper.clean_wd()
    wrapper.run(link_basis=basis)
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

    logging.info(" --- CO2 MP2 HESSIAN VERIFICATION USING BASIS={} --- ".format(basis))
    wrapper = Wrapper(wd="2_co2_opt", inpfname="2_co2_hess.fly")

    wrapper.load_inpfile()
    wrapper.set_options({
        "contrl" : {"SCFTYP": "RHF", "MPLEVL": 2, "RUNTYP": "HESSIAN", "MULT": 1 , "UNITS": "BOHR",
                    "ICUT": 11, "INTTYP": "HONDO", "MAXIT": 100},
        "basis"  : {"GBASIS": basis, "EXTFILE": ".T."},
        "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "NCONV": 8, "ENGTHR": 9, "FDIFF": ".F."},
        "mp2"    : {"METHOD": 1},
        "data"   : {"COMMENT": "CO2 HESSIAN AT OPT", "SYMMETRY": "C1", "GEOMETRY": opt},
        "force"  : {"NVIB" : 2, "PROJCT" : ".T."},
    })
    wrapper.save_inpfile("2_co2_mp2_hess-basis={}.fly".format(basis.lower()))

    wrapper.clean_wd()
    wrapper.run(link_basis=basis)
    wrapper.clean_up()

    wrapper.load_out()
    freqs = wrapper.frequencies()

    logging.info("Frequencies at optimized geometry (cm-1):")
    for f in freqs:
        logging.info("  {:.3f}".format(f))

    positive = all(f >= 0.0 for f in freqs)
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

    t = T / 1000.0
    return A + B * t + C * t**2 + D * t**3 + E * t**(-2) # J/(mol*K)

def NIST_CO2_S(T):
    A = 24.99735
    B = 55.18696
    C = -33.69137
    D = 7.948387
    E = -0.136638
    G = 228.2431

    t = T / 1000.0
    return A * np.log(t) + B * t + C * t**2 / 2 + D * t**3 / 3 - E / (2.0 * t**2) + G # J/mol/K


def run_cp_vib():
    gbasis = 'CC-PVDZ'
    wrapper = Wrapper(wd="3_co2_thermo", inpfname=f"3_co2_mp2_thermo-basis={gbasis}.fly")
    wrapper.load_out()

    T = np.linspace(0.01, 3000.0, 300)

    Cpvib_q  = np.asarray([wrapper.vib_cp_q(tt) for tt in T])
    Cpvib_cl = np.asarray([wrapper.vib_cp_cl(tt) for tt in T])

    plt.figure(figsize=(10, 8))

    plt.plot(T, Cpvib_cl, color='#FF6F61', linestyle='--', label='Class')
    plt.plot(T, Cpvib_q, color='#CFBFF7', label='Q')

    plt.xlim((0.0, 3000.0))
    plt.ylim((0.0, 35.0))

    plt.xlabel("Temperature, K")
    plt.ylabel(r"$C_p^\textrm{vib}$, J $\cdot$ mol$^{-1} \cdot$K$^{-1}$")

    plt.legend(fontsize=14)

    plt.show()

def run_entropy_rot():
    gbasis = 'CC-PVTZ'
    wrapper = Wrapper(wd="3_co2_thermo", inpfname=f"3_co2_mp2_thermo-basis={gbasis}.fly")
    wrapper.load_out()

    T = np.linspace(0.01, 5.0, 100)

    rot_S_q  = np.asarray([wrapper.rot_entropy_q(tt) for tt in T])
    rot_S_cl = np.asarray([wrapper.rot_entropy_cl(tt) for tt in T])

    plt.figure(figsize=(10, 8))

    plt.plot(T, rot_S_cl, color='#FF6F61', linestyle='--', label='Class')
    plt.plot(T, rot_S_q, color='#CFBFF7', label='Q')

    plt.show()

def run_cp_rot():
    gbasis = 'CC-PVDZ'
    wrapper = Wrapper(wd="3_co2_thermo", inpfname=f"3_co2_mp2_thermo-basis={gbasis}.fly")
    wrapper.load_out()

    #qrot_cl = wrapper.rot_partf_cl(T=200.0)
    #qrot_q = wrapper.rot_partf_q(T=200.0)

    #print("[Q]  Rotational partf: {}".format(qrot_q))
    #print("[Cl] Classical  partf: {}".format(qrot_cl))

    #Urot_cl = wrapper.rot_int_energy_cl(T=200.0)
    #Urot_q  = wrapper.rot_int_energy_q(T=200.0)

    #print("[Q]  Rotational U: {}".format(Urot_q))
    #print("[Cl] Rotational U: {}".format(Urot_cl))

    T = np.linspace(0.01, 10.0, 300)

    Cprot_cl = np.asarray([wrapper.rot_cp_cl(tt) for tt in T])
    Cprot_q  = np.asarray([wrapper.rot_cp_q(tt) for tt in T])

    Cprot_EM1 = np.asarray([wrapper.rot_cp_EM1(tt) for tt in T])
    Cprot_EM2 = np.asarray([wrapper.rot_cp_EM2(tt) for tt in T])
    Cprot_EM3 = np.asarray([wrapper.rot_cp_EM3(tt) for tt in T])
    Cprot_EM4 = np.asarray([wrapper.rot_cp_EM4(tt) for tt in T])

    plt.figure(figsize=(10, 8))

    plt.plot(T, Cprot_cl, color='#FF6F61', linestyle='--', label='Class')
    plt.plot(T, Cprot_q,  color='#CFBFF7', label='QM')
    #plt.plot(T, Cprot_EM1, color='#6CD4FF', label='EM-1')
    #plt.plot(T, Cprot_EM2, color='#88B04B', label='EM-2')
    #plt.plot(T, Cprot_EM3, color='#6B5B95', label='EM-3')
    #plt.plot(T, Cprot_EM4, color='yellow', label='EM-4')

    plt.xlabel(r"Temperature, K")
    plt.ylabel(r"$C_p^\textrm{rot}$, J $\cdot$ mol$^{-1} \cdot$K$^{-1}$")

    plt.xlim((0.0, 6.0))
    plt.ylim((0.0, 15.0))

    plt.legend(fontsize=14)

    plt.show()


def run_example_03():
    gbasis = 'CC-PVTZ'

    #wrapper = Wrapper(wd="3_co2_thermo", inpfname=f"3_co2_mp2_thermo-basis={gbasis}.fly")
    #wrapper.load_out()

    #logging.info(f" --- CO2 RHF THERMOCHEMISTRY USING BASIS={gbasis} --- ")
    #wrapper = Wrapper(wd="3_co2_thermo", inpfname="3_co2_thermo.fly")

    #wrapper.load_inpfile()

    #temperatures = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    #temperature_field = ", ".join(list(map(str, temperatures)))

    #wrapper.set_options({
    #    "contrl" : {"SCFTYP": "RHF", "MAXIT": 100, "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR",
    #                "ICUT": 11, "INTTYP": "HONDO", "MAXIT": 100},
    #    "system" : {"memory" : 12000000},
    #    "basis"  : {"GBASIS": basis, "EXTFILE": ".T."},
    #    "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "NCONV": 8, "ENGTHR": 9, "FDIFF": ".F."},
    #    "statpt" : {"METHOD": "NR", "UPHESS": "BFGS", "OPTTOL": 1e-5, "HSSEND": ".T."},
    #    "force" : {"TEMP(1)" : temperature_field}
    #})
    #wrapper.save_inpfile(f"3_co2_rhf_thermo-basis={gbasis}.fly")

    #wrapper.clean_wd()
    #wrapper.run(link_basis=gbasis)
    #wrapper.clean_up()

    #wrapper.load_out()
    #thermo = wrapper.thermo()

    #Cp_RHF = [block["CP"] for _, block in thermo.items()]
    #print("Cp_RHF: {}".format(Cp_RHF))

    #############################################################################

    logging.info(f" --- CO2 MP2 THERMOCHEMISTRY USING BASIS={gbasis} --- ")
    wrapper = Wrapper(wd="3_co2_thermo", inpfname="3_co2_thermo.fly")

    wrapper.load_inpfile()

    temperatures = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    temperature_field = ", ".join(list(map(str, temperatures)))

    wrapper.set_options({
        "contrl": {"SCFTYP": "RHF", "MPLEVL": 2, "RUNTYP": "OPTIMIZE", "MULT": 1, "UNITS": "BOHR",
                   "ICUT": 11, "INTTYP": "HONDO", "MAXIT": 100},
        "system" : {"memory" : 12000000},
        "basis" : {"GBASIS": gbasis, "EXTFILE": ".T."},
        "mp2"   : {"METHOD": 1},
        "statpt" : {"METHOD": "NR", "UPHESS": "BFGS", "OPTTOL": 1e-5, "HSSEND": ".T."},
        "scf"   : {"DIRSCF": ".T.", "DIIS": ".T.", "NCONV": 8, "ENGTHR": 9, "FDIFF": ".F."},
        "force" : {"TEMP(1)" : temperature_field}
    })
    wrapper.save_inpfile(f"3_co2_mp2_thermo-basis={gbasis}.fly")

    #wrapper.clean_wd()
    #wrapper.run(link_basis=gbasis)
    #wrapper.clean_up()

    wrapper.load_out()
    thermo = wrapper.thermo()

    Cp_MP2 = [block["CP"] * CAL_TO_J for _, block in thermo.items()]

    #############################################################################

    def tot_entropy(T):
        trans_entropy = wrapper.trans_entropy(T)
        rot_entropy   = wrapper.rot_entropy_cl(T)
        vib_entropy   = wrapper.vib_entropy_q(T)
        return trans_entropy + rot_entropy + vib_entropy

    temperatures = np.linspace(200.0, 1000.0, 300)

    S_MP2 = np.asarray([tot_entropy(tt) for tt in temperatures])
    S_NIST = np.asarray([NIST_CO2_S(tt) for tt in temperatures])

    plt.figure(figsize=(10, 10))

    plt.plot(temperatures, S_MP2, color='#FF6F61', label=f"MP2/{gbasis}")
    plt.plot(temperatures, S_NIST, color='#CFBFF7', label="NIST", linestyle='--')

    plt.xlabel("Temperature")
    plt.ylabel(r"S, J$\cdot$mol$^{-1}\cdot$K$^{-1}$")

    plt.legend(fontsize=14)

    plt.show()

    #def tot_cp(T):
    #    trans_cp  = wrapper.trans_cp(T)
    #    rot_cp_q = wrapper.rot_cp_q(T)
    #    vib_cp_q  = wrapper.vib_cp_q(T)
    #    total_cp = trans_cp + rot_cp_q + vib_cp_q
    #    return total_cp


    #temperatures = np.linspace(1.0, 50.0, 100)
    #Cp_MP2  = np.asarray([tot_cp(tt) for tt in temperatures])
    #Cp_NIST = np.asarray([NIST_CO2_CP(tt) for tt in temperatures])

    #plt.figure(figsize=(10, 10))

    #plt.plot(temperatures, Cp_MP2, color='#FF6F61', label=f"MP2/{gbasis}")

    #plt.legend(fontsize=14)
    #plt.xlabel("Temperature, K")
    #plt.ylabel("$C_p$, J$\cdot$mol$^{-1} \cdot$ K$^{-1}$")

    #plt.show()

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

def basis_extrapolation():
    n  = np.array([2.0, 3.0, 4.0])

    mode = 'bond'

    if mode == 'nu2':
        freqs = np.array([645.66, 658.94, 662.65])
    elif mode == 'nu1':
        freqs = np.array([1327.69, 1331.60, 1334.69])
    elif mode == 'nu3':
        freqs = np.array([2442.93, 2422.67, 2420.14])
    elif mode == 'bond':
        freqs = np.array([1.176, 1.169, 1.166])

    def model(x, e_cbs, a):
        return e_cbs + a * x**(-3)

    from scipy.optimize import curve_fit

    if mode == 'nu2':
        initial_params = np.array([670.0, -150.0])
    elif mode == 'nu1':
        initial_params = np.array([1350.0, -50.0])
    elif mode == 'nu3':
        initial_params = np.array([2400.0, 50.0])
    elif mode == 'bond':
        initial_params = np.array([1.160, 1.0])

    popt, pcov = curve_fit(model, n, freqs, p0=initial_params, method='lm', epsfcn=1e-3)

    print("Optimized parameters: {}".format(popt))

    n_min, n_max = 2.0, 4.0
    n_pred  = np.linspace(n_min, n_max, 300)
    freqs_pred = model(n_pred, *popt)

    plt.figure(figsize=(10, 8))

    plt.scatter(n, freqs, color='r', s=40)
    plt.plot(n_pred, freqs_pred, color='y', lw=2.0)

    if mode == 'nu2':
        plt.ylim((600.0, 700.0))
    elif mode == 'nu1':
        plt.ylim((1300.0, 1350.0))
    elif mode == 'nu3':
        plt.ylim((2400.0, 2450.0))
    elif mode == 'bond':
        plt.ylim((1.160, 1.180))

    plt.show()



if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    run_cp_rot()
    #run_cp_vib()

    #basis_extrapolation()

    #run_example_01()
    #run_example_02()
    #run_example_03()
    #run_example_04()



