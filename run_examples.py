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

BOHR_TO_ANG   = 0.529177
CAL_TO_J      = 4.184
H_TO_KCAL_MOL = 627.509608

Atom = namedtuple("Atom", ["symbol", "charge", "x", "y", "z"])

class Wrapper:
    EXE     = "/home/artfin/Desktop/QC-2022/firefly/firefly820"
    PROCGRP = "/home/artfin/Desktop/QC-2022/firefly/procgrp"
    P2PLIB  = "/home/artfin/Desktop/QC-2022/firefly/"
    LIB     = "/home/artfin/Desktop/QC-2022/firefly-examples/lib/"

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

    @classmethod
    def generate_input(cls, wd, inpfname, options):
        wd = os.path.abspath(wd)
        fname = os.path.join(wd, inpfname)
        open(fname, 'w').close()

        obj = cls(wd, inpfname)

        obj.inp_code = [""]
        obj.set_options(options)
        obj.inp_code = obj.inp_code[::-1]

        obj.save_inpfile(fname)

        return obj

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
        return "\n".join(textwrap.wrap(s, width=WIDTH, initial_indent='', replace_whitespace=True))

    def set_block(self, group_name, options):
        group_start, group_end = self.locate_group(self.inp_code, group=group_name)
        if group_start is not None and group_end is not None:
            del self.inp_code[group_start : group_end + 1]
        else:
            group_start, _ = self.locate_group(self.inp_code, group="$DATA")
            if group_start is None:
                group_start = 0

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
        for n in range(4):
            tmp_dir = self.wd + f'.{n}'
            if os.path.isdir(tmp_dir):
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
        cmd = f"{self.EXE} -r -f -p -stdext -i {self.inpfname} -o {self.outfname} -t {self.wd} -ex {self.P2PLIB} -p4pg {self.PROCGRP}"

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
        elif method == "optimize":
            pattern = "TOTAL ENERGY = + ([-+]?\d+.\d+)"
            energy = re.findall(pattern, lines)
            energy = list(map(float, energy))
        elif method == "solution":
            pattern = "TOTAL FREE ENERGY IN SOLVENT + = + ([-+]?\d+.\d+)"
            energy = re.findall(pattern, lines)
            assert len(energy) == 1
            energy = float(energy[0])
        else:
            raise ValueError("unreachable")

        return energy

    def parse_zpe(self):
        lines = "".join(self.outfile)
        pattern = "THE HARMONIC ZERO POINT ENERGY IS \(SCALED BY   1.000\)\n + (\d+.\d+) HARTREE/MOLECULE"
        zpe = re.findall(pattern, lines)
        assert len(zpe) == 1
        return float(zpe[0])

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
 ELEC. + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+
 TRANS. + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+
 ROT. + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+
 VIB. + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+ + [-+]?\d+.\d+
 TOTAL +([-+]?\d+.\d+) +([-+]?\d+.\d+) +([-+]?\d+.\d+) +([-+]?\d+.\d+) +([-+]?\d+.\d+) +([-+]?\d+.\d+)"""

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

    basis = "CC-PVTZ"
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
    gbasis = 'CC-PVTZ'
    wrapper = Wrapper(wd="3_co2_thermo", inpfname=f"3_co2_mp2_thermo-basis={gbasis}.fly")
    wrapper.load_out()

    qrot_cl = wrapper.rot_partf_cl(T=1.0)
    qrot_q = wrapper.rot_partf_q(T=1.0)

    print("[Q]  Rotational partf: {}".format(qrot_q))
    print("[Cl] Classical  partf: {}".format(qrot_cl))

    Urot_cl = wrapper.rot_int_energy_cl(T=1.0)
    Urot_q  = wrapper.rot_int_energy_q(T=1.0)

    print("[Q]  Rotational U: {}".format(Urot_q))
    print("[Cl] Rotational U: {}".format(Urot_cl))

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


def nitrobenzene():
    # INITIAL GEOMETRY
    #  Atom(symbol="C", charge=6, x=-3.5048225421, y= 0.0711805817, z= 0.1456897967),
    #  Atom(symbol="C", charge=6, x=-2.1242069042, y= 0.0676957680, z= 0.1437250554),
    #  Atom(symbol="C", charge=6, x=-1.4565144627, y= 1.2657898054, z= 0.0112805274),
    #  Atom(symbol="C", charge=6, x=-2.1243502782, y= 2.4616659201, z=-0.1394727314),
    #  Atom(symbol="C", charge=6, x=-3.5049153121, y= 2.4578370923, z=-0.1457245349),
    #  Atom(symbol="C", charge=6, x=-4.1936081427, y= 1.2645153194, z= 0.0001955136),
    #  Atom(symbol="H", charge=1, x=-4.0381801262, y=-0.8505059514, z= 0.2559173303),
    #  Atom(symbol="H", charge=1, x=-1.5620288767, y=-0.8346363876, z= 0.2389155097),
    #  Atom(symbol="H", charge=1, x=-1.5619534389, y= 3.3630228735, z=-0.2428628637),
    #  Atom(symbol="H", charge=1, x=-4.0382012347, y= 3.3785626398, z=-0.2639829256),
    #  Atom(symbol="H", charge=1, x=-5.2650389640, y= 1.2641688843, z=-0.0022762561),
    #  Atom(symbol="N", charge=7, x=-0.0085078655, y= 1.2648596634, z=-0.0056641832),
    #  Atom(symbol="O", charge=8, x= 0.5639468379, y= 0.1670702678, z=-0.1297708787),
    #  Atom(symbol="O", charge=8, x= 0.5668300231, y= 2.3598431617, z= 0.1306822195),

    # Optimized geometry at B3LYP/aug-cc-pvdz [takes almost 3 hours of optimization from the INITIAL 
    # geometry above]

    def nitrobenzene_optimization():
        gbasis = "ACC-PVDZ"
        logging.info(f" --- NITROBENZENE DFT ENERGY USING BASIS={gbasis} --- ")
        geom = [
            Atom(symbol="C", charge=6.0, x=-3.5182554149, y= 0.0562214119, z= 0.1169962360),
            Atom(symbol="C", charge=6.0, x=-2.1225554325, y= 0.0475375150, z= 0.1182080376),
            Atom(symbol="C", charge=6.0, x=-1.4488934552, y= 1.2642914380, z=-0.0003702357),
            Atom(symbol="C", charge=6.0, x=-2.1223665235, y= 2.4810881214, z=-0.1197323600),
            Atom(symbol="C", charge=6.0, x=-3.5180686681, y= 2.4726126865, z=-0.1186804612),
            Atom(symbol="C", charge=6.0, x=-4.2147274028, y= 1.2645070081, z=-0.0004746888),
            Atom(symbol="H", charge=1.0, x=-4.0625666258, y=-0.8829026850, z= 0.2078697524),
            Atom(symbol="H", charge=1.0, x=-1.5543436055, y=-0.8744126626, z= 0.2082094593),
            Atom(symbol="H", charge=1.0, x=-1.5540485677, y= 3.4029710264, z=-0.2096948047),
            Atom(symbol="H", charge=1.0, x=-4.0622624422, y= 3.4117648649, z=-0.2101477738),
            Atom(symbol="H", charge=1.0, x=-5.3046313610, y= 1.2646613789, z= 0.0002413395),
            Atom(symbol="N", charge=7.0, x= 0.0293020707, y= 1.2642495395, z= 0.0009854496),
            Atom(symbol="O", charge=8.0, x= 0.6007950737, y= 0.1830642665, z= 0.1150639960),
            Atom(symbol="O", charge=8.0, x= 0.6010710678, y= 2.3454157290, z=-0.1118223671),
        ]

        options = {
            "contrl" : {"SCFTYP": "RHF", "DFTTYP": "B3LYP", "RUNTYP" : "OPTIMIZE", "MULT": 1, "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
            "system" : {"memory" : 12000000},
            "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
            "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
            "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F.", "NCONV": 8, "ENGTHR": 9},
            "statpt" : {"METHOD": "GDIIS", "UPHESS": "BFGS"},
            "data"   : {"COMMENT": "NITROBENZENE", "SYMMETRY": "C1", "GEOMETRY": geom},
        }

        wrapper = Wrapper.generate_input(wd="nitrobenzene", inpfname="nitrobenzene.fly", options=options)

        wrapper.clean_wd()
        wrapper.run(link_basis=gbasis)
        wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method="rhf")
        logging.info("DFT B3LYP ENERGY: {}".format(energy))
        logging.info("---------------------------------------------------------\n")

    def nitrobenzene_freqs():
        gbasis = "ACC-PVDZ"
        logging.info(f" --- NITROBENZENE DFT HESSIAN USING BASIS={gbasis} --- ")
        geom = [
            Atom(symbol="C", charge=6.0, x=-3.5182554149, y= 0.0562214119, z= 0.1169962360),
            Atom(symbol="C", charge=6.0, x=-2.1225554325, y= 0.0475375150, z= 0.1182080376),
            Atom(symbol="C", charge=6.0, x=-1.4488934552, y= 1.2642914380, z=-0.0003702357),
            Atom(symbol="C", charge=6.0, x=-2.1223665235, y= 2.4810881214, z=-0.1197323600),
            Atom(symbol="C", charge=6.0, x=-3.5180686681, y= 2.4726126865, z=-0.1186804612),
            Atom(symbol="C", charge=6.0, x=-4.2147274028, y= 1.2645070081, z=-0.0004746888),
            Atom(symbol="H", charge=1.0, x=-4.0625666258, y=-0.8829026850, z= 0.2078697524),
            Atom(symbol="H", charge=1.0, x=-1.5543436055, y=-0.8744126626, z= 0.2082094593),
            Atom(symbol="H", charge=1.0, x=-1.5540485677, y= 3.4029710264, z=-0.2096948047),
            Atom(symbol="H", charge=1.0, x=-4.0622624422, y= 3.4117648649, z=-0.2101477738),
            Atom(symbol="H", charge=1.0, x=-5.3046313610, y= 1.2646613789, z= 0.0002413395),
            Atom(symbol="N", charge=7.0, x= 0.0293020707, y= 1.2642495395, z= 0.0009854496),
            Atom(symbol="O", charge=8.0, x= 0.6007950737, y= 0.1830642665, z= 0.1150639960),
            Atom(symbol="O", charge=8.0, x= 0.6010710678, y= 2.3454157290, z=-0.1118223671),
        ]

        options = {
            "contrl" : {"SCFTYP": "RHF", "DFTTYP": "B3LYP", "RUNTYP" : "HESSIAN", "MULT": 1, "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
            "system" : {"memory" : 12000000},
            "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
            "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
            "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F.", "NCONV": 8, "ENGTHR": 9},
            "force"  : {"NVIB" : 1, "PROJCT": ".T."},
            "data"   : {"COMMENT": "NITROBENZENE", "SYMMETRY": "C1", "GEOMETRY": geom},
        }

        wrapper = Wrapper.generate_input(wd="nitrobenzene", inpfname="nitrobenzene-hessian.fly", options=options)

        wrapper.clean_wd()
        wrapper.run(link_basis=gbasis)
        wrapper.clean_up()

        wrapper.load_out()
        logging.info("---------------------------------------------------------\n")


    def nitrobenzene_energy_pcm(run=True):
        gbasis = "ACC-PVDZ"
        logging.info(f" --- NITROBENZENE DFT/PCM ENERGY USING BASIS={gbasis} --- ")

        geom = [
           Atom(symbol="C", charge=6.0, x=-3.5182554149, y= 0.0562214119, z= 0.1169962360),
           Atom(symbol="C", charge=6.0, x=-2.1225554325, y= 0.0475375150, z= 0.1182080376),
           Atom(symbol="C", charge=6.0, x=-1.4488934552, y= 1.2642914380, z=-0.0003702357),
           Atom(symbol="C", charge=6.0, x=-2.1223665235, y= 2.4810881214, z=-0.1197323600),
           Atom(symbol="C", charge=6.0, x=-3.5180686681, y= 2.4726126865, z=-0.1186804612),
           Atom(symbol="C", charge=6.0, x=-4.2147274028, y= 1.2645070081, z=-0.0004746888),
           Atom(symbol="H", charge=1.0, x=-4.0625666258, y=-0.8829026850, z= 0.2078697524),
           Atom(symbol="H", charge=1.0, x=-1.5543436055, y=-0.8744126626, z= 0.2082094593),
           Atom(symbol="H", charge=1.0, x=-1.5540485677, y= 3.4029710264, z=-0.2096948047),
           Atom(symbol="H", charge=1.0, x=-4.0622624422, y= 3.4117648649, z=-0.2101477738),
           Atom(symbol="H", charge=1.0, x=-5.3046313610, y= 1.2646613789, z= 0.0002413395),
           Atom(symbol="N", charge=7.0, x= 0.0293020707, y= 1.2642495395, z= 0.0009854496),
           Atom(symbol="O", charge=8.0, x= 0.6007950737, y= 0.1830642665, z= 0.1150639960),
           Atom(symbol="O", charge=8.0, x= 0.6010710678, y= 2.3454157290, z=-0.1118223671),
        ]

       # UFF model: 
       # A. K. Rappe, C. J. Casewit, K. S. Colwell, W. A. Goddard, and W. M. Skiff. UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations. J. Am. Chem. Soc., 114(25):10024–10035, 1992. URL: http://pubs.acs.org/doi/abs/10.1021/ja00051a040, doi:10.1021/ja00051a040.

       #xe = (-3.5182554149, -2.1225554325, -1.4488934552, -2.1223665235, -3.5180686681, -4.2147274028, -4.0625666258,
       #      -1.5543436055, -1.5540485677, -4.0622624422, -5.3046313610, 0.0293020707, 0.6007950737, 0.6010710678)
       #ye = (0.0562214119, 0.0475375150, 1.2642914380, 2.4810881214, 2.4726126865, 1.2645070081, -0.8829026850,
       #     -0.8744126626, 3.4029710264, 3.4117648649, 1.2646613789, 1.2642495395, 0.1830642665, 2.3454157290)
       #ze = (0.1169962360, 0.1182080376, -0.0003702357, -0.1197323600, -0.1186804612, -0.0004746888, 0.2078697524,
       #      0.2082094593, -0.2096948047, -0.2101477738, 0.0002413395, 0.0009854496, 0.1150639960, -0.1118223671)

       #radii = (1.9255,) * 6 + (1.4430,) * 5 + (1.83,) + (1.75,) * 2

       #xe_field = ", ".join(list(map(str, xe)))
       #ye_field = ", ".join(list(map(str, ye)))
       #ze_field = ", ".join(list(map(str, ze)))
       #RIN_field = ", ".join(list(map(str, radii)))

        options = {
            "contrl" : {"SCFTYP": "RHF", "DFTTYP": "B3LYP", "MULT": 1, "UNITS": "ANGS",
                        "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
            "system" : {"memory" : 12000000},
            "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
            "pcm"    : {"PCMTYP" : "CPCM", "EPS": 36.64, "EPSINF": 1.806, "RSOLV": 2.155, "ICAV" : 1,
                       "TCE" : 0.001397, "VMOL" : 52.8521, "STEN": 29.29, "IDP": 1, "IREP" : 1},
            "newcav" : {"RHOW": 0.77674, "PM" : 41.0524, "NEVAL": 16},
            #"pcmcav" : {"XE(1)": xe_field, "YE(1)": ye_field, "ZE(1)": ze_field, "RIN(1)": RIN_field},
            "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
            "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F.", "NCONV": 8, "ENGTHR": 9},
            "data"   : {"COMMENT": "NITROBENZENE", "SYMMETRY": "C1", "GEOMETRY": geom},
        }

        wrapper = Wrapper.generate_input(wd="nitrobenzene", inpfname="nitrobenzene-ch3cn.fly", options=options)

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method="solution")
        logging.info("PCM ENERGY OF NITROBENZENE: {}".format(energy))
        logging.info("---------------------------------------------------------\n")

    def nitrobenzene_anion_optimization():
        gbasis = "ACC-PVDZ"
        logging.info(f" --- NITROBENZENE DFT ENERGY USING BASIS={gbasis} --- ")
        # geom = [
        #     Atom(symbol="C", charge=6.0, x=-3.5182554149, y= 0.0562214119, z= 0.1169962360),
        #     Atom(symbol="C", charge=6.0, x=-2.1225554325, y= 0.0475375150, z= 0.1182080376),
        #     Atom(symbol="C", charge=6.0, x=-1.4488934552, y= 1.2642914380, z=-0.0003702357),
        #     Atom(symbol="C", charge=6.0, x=-2.1223665235, y= 2.4810881214, z=-0.1197323600),
        #     Atom(symbol="C", charge=6.0, x=-3.5180686681, y= 2.4726126865, z=-0.1186804612),
        #     Atom(symbol="C", charge=6.0, x=-4.2147274028, y= 1.2645070081, z=-0.0004746888),
        #     Atom(symbol="H", charge=1.0, x=-4.0625666258, y=-0.8829026850, z= 0.2078697524),
        #     Atom(symbol="H", charge=1.0, x=-1.5543436055, y=-0.8744126626, z= 0.2082094593),
        #     Atom(symbol="H", charge=1.0, x=-1.5540485677, y= 3.4029710264, z=-0.2096948047),
        #     Atom(symbol="H", charge=1.0, x=-4.0622624422, y= 3.4117648649, z=-0.2101477738),
        #     Atom(symbol="H", charge=1.0, x=-5.3046313610, y= 1.2646613789, z= 0.0002413395),
        #     Atom(symbol="N", charge=7.0, x= 0.0293020707, y= 1.2642495395, z= 0.0009854496),
        #     Atom(symbol="O", charge=8.0, x= 0.6007950737, y= 0.1830642665, z= 0.1150639960),
        #     Atom(symbol="O", charge=8.0, x= 0.6010710678, y= 2.3454157290, z=-0.1118223671),
        # ]

        # FINAL OPTIMIZED GEOMETRY OF THE ANION
        geom = [
            Atom(symbol="C", charge=6.0, x=-3.5161599401, y= 0.0593660329, z= 0.1168269711), 
            Atom(symbol="C", charge=6.0, x=-2.1253899288, y= 0.0456644961, z= 0.1187617775),
            Atom(symbol="C", charge=6.0, x=-1.4021428638, y= 1.2642993454, z=-0.0001776956),
            Atom(symbol="C", charge=6.0, x=-2.1251816210, y= 2.4829698318, z=-0.1200130153),
            Atom(symbol="C", charge=6.0, x=-3.5159561698, y= 2.4694691969, z=-0.1183486459),
            Atom(symbol="C", charge=6.0, x=-4.2365489151, y= 1.2644941339, z=-0.0005988369),
            Atom(symbol="H", charge=1.0, x=-4.0562222427, y=-0.8863760719, z= 0.2085871997),
            Atom(symbol="H", charge=1.0, x=-1.5589149006, y=-0.8775282165, z= 0.2089504254),
            Atom(symbol="H", charge=1.0, x=-1.5585624145, y= 3.4060584081, z=-0.2103848504),
            Atom(symbol="H", charge=1.0, x=-4.0558668260, y= 3.4152597388, z=-0.2105029742),
            Atom(symbol="H", charge=1.0, x=-5.3275005778, y= 1.2646192515, z=-0.0002542196),
            Atom(symbol="N", charge=7.0, x=-0.0069871904, y= 1.2642654950, z= 0.0009040651),
            Atom(symbol="O", charge=8.0, x= 0.6168189949, y= 0.1448844400, z= 0.1174188059),
            Atom(symbol="O", charge=8.0, x= 0.6170633085, y= 2.3836235564, z=-0.1145174279),
        ]

        options = {
            "contrl" : {"SCFTYP": "UHF", "DFTTYP": "B3LYP", "RUNTYP" : "OPTIMIZE", "ICHARG": -1, "MULT": 2, "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
            "system" : {"memory" : 12000000},
            "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
            "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
            "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F.", "NCONV": 8, "ENGTHR": 9},
            "statpt" : {"METHOD": "GDIIS", "UPHESS": "BFGS"},
            "data"   : {"COMMENT": "NITROBENZENE", "SYMMETRY": "C1", "GEOMETRY": geom},
        }

        wrapper = Wrapper.generate_input(wd="nitrobenzene", inpfname="nitrobenzene-anion.fly", options=options)

        wrapper.clean_wd()
        wrapper.run(link_basis=gbasis)
        wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method="rhf")
        logging.info("DFT B3LYP ENERGY: {}".format(energy))
        logging.info("---------------------------------------------------------\n")

    def nitrobenzene_anion_freqs():
        gbasis = "ACC-PVDZ"
        logging.info(f" --- NITROBENZENE ANION-RADICAL DFT HESSIAN USING BASIS={gbasis} --- ")

        geom = [
            Atom(symbol="C", charge=6.0, x=-3.5161599401, y= 0.0593660329, z= 0.1168269711), 
            Atom(symbol="C", charge=6.0, x=-2.1253899288, y= 0.0456644961, z= 0.1187617775),
            Atom(symbol="C", charge=6.0, x=-1.4021428638, y= 1.2642993454, z=-0.0001776956),
            Atom(symbol="C", charge=6.0, x=-2.1251816210, y= 2.4829698318, z=-0.1200130153),
            Atom(symbol="C", charge=6.0, x=-3.5159561698, y= 2.4694691969, z=-0.1183486459),
            Atom(symbol="C", charge=6.0, x=-4.2365489151, y= 1.2644941339, z=-0.0005988369),
            Atom(symbol="H", charge=1.0, x=-4.0562222427, y=-0.8863760719, z= 0.2085871997),
            Atom(symbol="H", charge=1.0, x=-1.5589149006, y=-0.8775282165, z= 0.2089504254),
            Atom(symbol="H", charge=1.0, x=-1.5585624145, y= 3.4060584081, z=-0.2103848504),
            Atom(symbol="H", charge=1.0, x=-4.0558668260, y= 3.4152597388, z=-0.2105029742),
            Atom(symbol="H", charge=1.0, x=-5.3275005778, y= 1.2646192515, z=-0.0002542196),
            Atom(symbol="N", charge=7.0, x=-0.0069871904, y= 1.2642654950, z= 0.0009040651),
            Atom(symbol="O", charge=8.0, x= 0.6168189949, y= 0.1448844400, z= 0.1174188059),
            Atom(symbol="O", charge=8.0, x= 0.6170633085, y= 2.3836235564, z=-0.1145174279),
        ]

        options = {
            "contrl" : {"SCFTYP": "UHF", "DFTTYP": "B3LYP", "RUNTYP" : "HESSIAN", "ICHARG": -1, "MULT": 2, "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
            "system" : {"memory" : 12000000},
            "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
            "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
            "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F.", "NCONV": 8, "ENGTHR": 9},
            "force"  : {"NVIB" : 1, "PROJCT": ".T."},
            "data"   : {"COMMENT": "NITROBENZENE", "SYMMETRY": "C1", "GEOMETRY": geom},
        }

        wrapper = Wrapper.generate_input(wd="nitrobenzene", inpfname="nitrobenzene-anion-hessian.fly", options=options)

        wrapper.clean_wd()
        wrapper.run(link_basis=gbasis)
        wrapper.clean_up()

        wrapper.load_out()
        logging.info("---------------------------------------------------------\n")

    def nitrobenzene_anion_energy_pcm(run=True):
        gbasis = "ACC-PVDZ"
        logging.info(f" --- NITROBENZENE DFT/PCM ENERGY USING BASIS={gbasis} --- ")

        geom = [
            Atom(symbol="C", charge=6.0, x=-3.5161599401, y= 0.0593660329, z= 0.1168269711),
            Atom(symbol="C", charge=6.0, x=-2.1253899288, y= 0.0456644961, z= 0.1187617775),
            Atom(symbol="C", charge=6.0, x=-1.4021428638, y= 1.2642993454, z=-0.0001776956),
            Atom(symbol="C", charge=6.0, x=-2.1251816210, y= 2.4829698318, z=-0.1200130153),
            Atom(symbol="C", charge=6.0, x=-3.5159561698, y= 2.4694691969, z=-0.1183486459),
            Atom(symbol="C", charge=6.0, x=-4.2365489151, y= 1.2644941339, z=-0.0005988369),
            Atom(symbol="H", charge=1.0, x=-4.0562222427, y=-0.8863760719, z= 0.2085871997),
            Atom(symbol="H", charge=1.0, x=-1.5589149006, y=-0.8775282165, z= 0.2089504254),
            Atom(symbol="H", charge=1.0, x=-1.5585624145, y= 3.4060584081, z=-0.2103848504),
            Atom(symbol="H", charge=1.0, x=-4.0558668260, y= 3.4152597388, z=-0.2105029742),
            Atom(symbol="H", charge=1.0, x=-5.3275005778, y= 1.2646192515, z=-0.0002542196),
            Atom(symbol="N", charge=7.0, x=-0.0069871904, y= 1.2642654950, z= 0.0009040651),
            Atom(symbol="O", charge=8.0, x= 0.6168189949, y= 0.1448844400, z= 0.1174188059),
            Atom(symbol="O", charge=8.0, x= 0.6170633085, y= 2.3836235564, z=-0.1145174279),
        ]

        # UFF model: 
        # A. K. Rappe, C. J. Casewit, K. S. Colwell, W. A. Goddard, and W. M. Skiff. UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations. J. Am. Chem. Soc., 114(25):10024–10035, 1992. URL: http://pubs.acs.org/doi/abs/10.1021/ja00051a040, doi:10.1021/ja00051a040.

        #xe = (-3.5182554149, -2.1225554325, -1.4488934552, -2.1223665235, -3.5180686681, -4.2147274028, -4.0625666258,
        #      -1.5543436055, -1.5540485677, -4.0622624422, -5.3046313610, 0.0293020707, 0.6007950737, 0.6010710678)
        #ye = (0.0562214119, 0.0475375150, 1.2642914380, 2.4810881214, 2.4726126865, 1.2645070081, -0.8829026850,
        #     -0.8744126626, 3.4029710264, 3.4117648649, 1.2646613789, 1.2642495395, 0.1830642665, 2.3454157290)
        #ze = (0.1169962360, 0.1182080376, -0.0003702357, -0.1197323600, -0.1186804612, -0.0004746888, 0.2078697524,
        #      0.2082094593, -0.2096948047, -0.2101477738, 0.0002413395, 0.0009854496, 0.1150639960, -0.1118223671)

        #radii = (1.9255,) * 6 + (1.4430,) * 5 + (1.83,) + (1.75,) * 2

        #xe_field = ", ".join(list(map(str, xe)))
        #ye_field = ", ".join(list(map(str, ye)))
        #ze_field = ", ".join(list(map(str, ze)))
        #RIN_field = ", ".join(list(map(str, radii)))

        options = {
            "contrl" : {"SCFTYP": "UHF", "DFTTYP": "B3LYP", "ICHARG": -1, "MULT": 2, "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
            "system" : {"memory" : 12000000},
            "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
            "pcm"    : {"PCMTYP" : "CPCM", "EPS": 36.64, "EPSINF": 1.806, "RSOLV": 2.155, "ICAV" : 1,
                        "TCE" : 0.001397, "VMOL" : 52.8521, "STEN": 29.29, "IDP": 1, "IREP" : 1},
            "newcav" : {"RHOW": 0.77674, "PM" : 41.0524, "NEVAL": 16},
            #"pcm"    : {"PCMTYP" : "CPCM", "EPS": 36.64, "EPSINF": 1.806, "RSOLV": 2.155},
            #"pcmcav" : {"XE(1)": xe_field, "YE(1)": ye_field, "ZE(1)": ze_field, "RIN(1)": RIN_field},
            "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
            "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F.", "NCONV": 8, "ENGTHR": 9},
            "statpt" : {"METHOD": "GDIIS", "UPHESS": "BFGS"},
            "data"   : {"COMMENT": "NITROBENZENE", "SYMMETRY": "C1", "GEOMETRY": geom},
        }

        wrapper = Wrapper.generate_input(wd="nitrobenzene", inpfname="nitrobenzene-anion-ch3cn.fly", options=options)

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method="solution")
        logging.info("PCM ENERGY OF NITROBENZENE ANION: {}".format(energy))
        logging.info("---------------------------------------------------------\n")

    def gas_phase_ionization():
        E_NB_neutral = -436.8199112585
        E_NB_anion   = -436.8638008151

        ZPE_neutral = 0.102718
        ZPE_anion   = 0.099631

        AU_TO_EV = 27.211
        E_neutral = E_NB_neutral + ZPE_neutral
        E_anion   = E_NB_anion   + ZPE_anion
        EA = (E_neutral - E_anion) * AU_TO_EV

        print("Gas phase ionization potential: {} eV".format(EA))

    def redox_potential():
        # SCF E + Delta G(solv) 
        # PURELY ELECTROSTATIC PCM
        # E_NB_neutral_solv = -436.8257920569
        # E_NB_anion_solv   = -436.9478811603

        # ELEC + NON-ELEC PCM
        E_NB_neutral_solv = -436.8300804902
        E_NB_anion_solv   = -436.9623161587

        ZPE_neutral = 0.102718
        ZPE_anion   = 0.099631

        G_corr_neutral = 0.0708299
        G_corr_anion   = 0.067418

        AU_TO_EV = 27.211
        G_NB_neutral = E_NB_neutral_solv + ZPE_neutral + G_corr_neutral
        G_NB_anion   = E_NB_anion_solv   + ZPE_anion   + G_corr_anion
        dGred = (- G_NB_neutral + G_NB_anion) * AU_TO_EV

        print("Delta G of reduction: {}".format(dGred))

        ne = 1.0
        F  = 1.0
        NHE = -4.43 # eV; NHE = Normal Hydrogen Electrode
        redox = -dGred / ne / F + NHE
        print("Redox potential: {}".format(redox))


    # Neutral molecule
    # Gas phase: DFT B3LYP ENERGY: -436.8199112585                                       [90 min optimization]
    # CH3CN:     DFT B3LYP ENERGY: -436.8257920569 <= UFF van der Waals atom radii
    # CH3CN:     DFT B3LYP ENERGY: -436.8288505477 <= default van der Waals atom radii

    # Hessian: [130 min x 4 proc]
    # THE HARMONIC ZERO POINT ENERGY IS 0.102718 HARTREE/MOLECULE  
    # free energy correction (total=trans+rot+vib): 44.446 kcal/mol = 0.0708299 Eh 

    # Anion
    # Gas phase: DFT B3LYP ENERGY: -436.8638008151                                       [60 min optimization from gas phase structure]
    # CH3CN:     DFT B3LYP ENERGY: -436.9478811603 <= UFF van der Waals atom radii

    # Hessian: [200 min x 4 proc]
    # THE HARMONIC ZERO POINT ENERGY IS 0.099631 HARTREE/MOLECULE
    # free energy correction (total=trans+rot+vib): 42.305 kcal/mol = 0.067418 Eh 

    # 1 Eh = 627.503 kcal/mol

    # charge | minimized E (gas) | free energy correction (gas) |      ZPE     | SCF E + Delta G(solv) 
    #    0   |  -436.8199112585  |           0.0708299          |   0.102718   |   -436.8257920569        
    #    -1  |  -436.8638008151  |           0.067418           |   0.099631   |   -436.9478811603  


    #gas_phase_ionization()
    #nitrobenzene_freqs()
    #nitrobenzene_anion_freqs()
    #nitrobenzene_optimization_pcm()
    #nitrobenzene_anion_optimization_pcm()

    nitrobenzene_energy_pcm(run=False)
    nitrobenzene_anion_energy_pcm(run=False)

    redox_potential()


def acetic_acid():
    gbasis = "ACC-PVDZ"
    logging.info(f" --- ACETIC ACID DFT ENERGY USING BASIS={gbasis} --- ")
    geom = [
        Atom(symbol="H", charge=1.0, x=-0.911466037, y= 0.134713529, z=-4.436015650),
        Atom(symbol="O", charge=8.0, x=-0.911466037, y=-0.373052567, z=-3.607012062),
        Atom(symbol="O", charge=8.0, x=-0.911466037, y= 1.737765752, z=-2.821469232),
        Atom(symbol="C", charge=6.0, x=-0.911466037, y= 0.546532271, z=-2.607978107),
        Atom(symbol="C", charge=6.0, x=-0.911466037, y=-0.113120930, z=-1.252643138),
        Atom(symbol="H", charge=1.0, x=-0.030318037, y=-0.753033297, z=-1.149291593),
        Atom(symbol="H", charge=1.0, x=-0.911466037, y= 0.650936769, z=-0.476687742),
        Atom(symbol="H", charge=1.0, x=-1.792614037, y=-0.753033297, z=-1.149291593),
    ]

    options = {
        "contrl" : {"SCFTYP": "RHF", "DFTTYP": "B3LYP", "RUNTYP" : "OPTIMIZE", "MULT": 1, "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
        "system" : {"memory" : 12000000},
        "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
        "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
        "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F."},
        "statpt" : {"METHOD": "GDIIS", "UPHESS": "BFGS"},
        "data"   : {"COMMENT": "ACETIC ACID", "SYMMETRY": "C1", "GEOMETRY": geom},
    }

    wrapper = Wrapper.generate_input(wd="acetic-acid", inpfname="acetic-acid.fly", options=options)

    wrapper.clean_wd()
    wrapper.run(link_basis=gbasis)
    wrapper.clean_up()

    logging.info("---------------------------------------------------------\n")

def dft_optimize(gbasis, charge, mult, geom):
    if mult == 1:
        SCFTYP = "RHF"
    elif mult == 2:
        SCFTYP = "UHF"

    return {
        "contrl" : {"SCFTYP": SCFTYP, "DFTTYP" : "B3LYP", "RUNTYP": "OPTIMIZE", "ICHARG": charge, "MULT" : mult,
                    "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 150},
        "system" : {"memory" : 12000000},
        "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
        "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
        "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F."},
        "statpt" : {"METHOD": "GDIIS", "UPHESS": "BFGS"},
        "data"   : {"COMMENT": "COMMENT", "SYMMETRY": "C1", "GEOMETRY": geom},
    }

def dft_hessian(gbasis, charge, mult, geom):
    if mult == 1:
        SCFTYP = "RHF"
    elif mult == 2:
        SCFTYP = "UHF"

    return {
        "contrl" : {"SCFTYP": SCFTYP, "DFTTYP" : "B3LYP", "RUNTYP" : "HESSIAN", "ICHARG" : charge, "MULT" : mult,
                    "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
        "system" : {"memory" : 12000000},
        "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
        "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
        "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F."},
        "force"  : {"NVIB" : 1, "PROJCT": ".T."},
        "data"   : {"COMMENT": "COMMENT", "SYMMETRY": "C1", "GEOMETRY": geom},
    }

Solvent = namedtuple("Solvent", ["EPS", "EPSINF", "RSOLV"])



def dft_optimize_pcm(gbasis, charge, geom, xe_field=None, ye_field=None, ze_field=None, RIN_field=None, solvent=None):
    return {
        "contrl" : {"SCFTYP": "RHF", "DFTTYP": "B3LYP", "RUNTYP": "OPTIMIZE", "ICHARG": charge,
                   "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
        "system" : {"memory" : 12000000},
        "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
#       "pcm"    : {"PCMTYP" : "CPCM", "EPS": solvent.EPS, "EPSINF": solvent.EPSINF, "RSOLV": solvent.RSOLV, "ICENT": 1},
#       "pcmcav" : {"XE(1)": xe_field, "YE(1)": ye_field, "ZE(1)": ze_field, "RIN(1)": RIN_field},
        "pcm"    : {"PCMTYP" : "DPCM", "solvnt" : "water"},
        "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
        "statpt" : {"METHOD": "GDIIS", "UPHESS": "BFGS"},
        "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F.", "NCONV": 8, "ENGTHR": 9},
        "data"   : {"COMMENT": "COMMENT", "SYMMETRY": "C1", "GEOMETRY": geom}
    }

def dft_hessian_pcm(gbasis, charge, geom):
    return {
       "contrl" : {"SCFTYP": "RHF", "DFTTYP": "B3LYP", "RUNTYP" : "HESSIAN", "ICHARG": charge,
                   "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
       "system" : {"memory" : 12000000},
       "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
       "force"  : {"NVIB" : 1, "PROJCT": ".T."},
       "pcm"    : {"PCMTYP" : "DPCM", "solvnt" : "water"},
       "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
       "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F.", "NCONV": 8, "ENGTHR": 9},
       "data"   : {"COMMENT": "COMMENT", "SYMMETRY": "C1", "GEOMETRY": geom}
    }

def dft_energy_pcm(gbasis, charge, mult, geom, xe_field=None, ye_field=None, ze_field=None, RIN_field=None, solvent=None):
    if mult == 1:
        SCFTYP = "RHF"
    elif mult == 2:
        SCFTYP = "UHF"

    # minimal exponent of the function of the given angular momentum in aug-cc-pVDZ basis set
    BASIS_MIN_EXPONENT = {
        "C" : {0 : 0.0469, 2 : 0.151},  #1 : 0.04041, 2 : 0.151},
        "H" : {0 : 0.02974}, #1 : 0.141},
        "O" : {0 : 0.07896, 2 : 0.332}, #1 : 0.06856, 2 : 0.332},
    }

    NKTYP = [l for atom in geom for l in list(BASIS_MIN_EXPONENT[atom.symbol].keys())]
    NKTYP_field = ", ".join(map(str, NKTYP))

    NADD = len(NKTYP)

    XYZE_field = ""
    for atom in geom:
        ls = BASIS_MIN_EXPONENT[atom.symbol]
        for (l, exponent) in ls.items():
            s = "{:.7f} {:.7f} {:.7f} {:.8f}\n".format(atom.x / BOHR_TO_ANG , atom.y / BOHR_TO_ANG, atom.z / BOHR_TO_ANG, exponent / 3) # NOTE: exponent is divided by 3
            XYZE_field += s

    return {
        "contrl" : {"SCFTYP": SCFTYP, "DFTTYP": "B3LYP", "ICHARG": charge, "MULT" : mult,
                   "UNITS": "ANGS", "INTTYP": "HONDO", "ICUT": 11, "ITOL": 30, "MAXIT": 100},
        "system" : {"memory" : 12000000},
        "p2p"    : {"P2P" : ".T.", "DLB" : ".T."},
        #"pcm"    : {"PCMTYP" : "CPCM", "ICAV": 1, "IDISP": 1, "solvnt" : "water"},
        #"disrep" : {"ICLAV" : 1, "RHO" : "3.348D-02", "N" : 2, "NT(1)" : "2,1", "RDIFF(1)": "1.20,1.50", "DKT(1)": "1.0,1.36", "RWT(1)": "1.2,1.5"},
        "pcm"    : {"PCMTYP" : "CPCM", "ICAV": 1, "IREP": 1, "IDP" : 1, "solvnt" : "water"},
        #"disbs"  : {"NADD" : NADD, "NKTYP(1)" : NKTYP_field, "XYZE(1)" : XYZE_field},
        "basis"  : {"GBASIS": gbasis, "EXTFILE": ".T."},
        "scf"    : {"DIRSCF": ".T.", "DIIS": ".T.", "FDIFF": ".F.", "NCONV": 8, "ENGTHR": 9},
        "data"   : {"COMMENT": "COMMENT", "SYMMETRY": "C1", "GEOMETRY": geom}
    }

def formic_acid():
    def free_energy_gas(run=True, charge=0):
        gbasis = "ACC-PVDZ"
        logging.info(f" --- FORMIC ACID DFT ENERGY USING BASIS={gbasis} CHARGE={charge} --- ")

        if charge == 0:
            geom = [
                Atom(symbol="C", charge=6.0, x=-0.136440, y= 0.398339, z=0.0),
                Atom(symbol="O", charge=8.0, x=-1.134475, y=-0.263170, z=0.0),
                Atom(symbol="O", charge=8.0, x=1.118039, y=-0.090345, z=0.0),
                Atom(symbol="H", charge=1.0, x=-0.097393, y=1.496365, z=0.0),
                Atom(symbol="H", charge=1.0, x=1.047524, y=-1.05278, z=0.0),
            ]
        elif charge == -1:
            geom = [
                Atom(symbol="H", charge=1.0, x=-0.000527, y=1.46308, z=0.0),
                Atom(symbol="C", charge=6.0, x=0.0, y=0.307752, z=0.0),
                Atom(symbol="O", charge=8.0, x=-1.137525, y=-0.207123, z=0.0),
                Atom(symbol="O", charge=8.0, x=1.137591, y=-0.206574, z=0.0),
            ]

        wrapper = Wrapper.generate_input(
            wd="formic-acid",
            inpfname="formic-acid-opt-charge={}.fly".format(charge),
            options=dft_optimize(gbasis, charge, mult=1, geom=geom)
        )

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method='optimize')[-1]
        print("Energy at optimized geometry: {} HARTREE".format(energy)) 

        geometries = wrapper.opt_geometries(natoms=len(geom))

        opt = geometries[-1]
        logging.info("Optimized geometry (ANG):")
        for atom in opt:
            logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")

        wrapper = Wrapper.generate_input(
            wd="formic-acid",
            inpfname="formic-acid-hess-charge={}.fly".format(charge),
            options=dft_hessian(gbasis, charge, mult=1, geom=opt)
        )

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        freqs = wrapper.frequencies()
        zpe = wrapper.parse_zpe()
        thermo = wrapper.thermo()

        positive = all(f >= 0.0 for f in freqs)
        logging.info("Assert freqs > 0: {}".format(positive))
        assert positive

        print("ZPE: {} HARTREE/MOLECULE".format(zpe))

        free_energy_corr = thermo[298.15]['G'] / H_TO_KCAL_MOL
        print("Free energy correction: {} HARTREE".format(free_energy_corr))

        free_energy = energy # + zpe + free_energy_corr
        print("Free energy: {} HARTREE".format(free_energy))

        logging.info("---------------------------------------------------------\n")

        return opt, free_energy

    def free_energy_solv(geom, run=True, charge=0):
        gbasis = "ACC-PVDZ"
        logging.info(f" --- FORMIC ACID DFT/PCM ENERGY USING BASIS={gbasis} charge={charge} --- ")

        # UFF model: 
        # A. K. Rappe, C. J. Casewit, K. S. Colwell, W. A. Goddard, and W. M. Skiff. UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations. J. Am. Chem. Soc., 114(25):10024–10035, 1992. URL: http://pubs.acs.org/doi/abs/10.1021/ja00051a040, doi:10.1021/ja00051a040.
        #UFF_RADII = {
        #    "C" : 1.9255,
        #    "H" : 1.4430,
        #    "N" : 1.83,
        #    "O" : 1.75
        #}

        #Source: https://pcmsolver.readthedocs.io/en/stable/users/input.html#available-solvents
        #WATER = Solvent(EPS=78.39, EPSINF=1.776, RSOLV=1.385)

        wrapper = Wrapper.generate_input(
            wd="formic-acid",
            inpfname="formic-acid-pcm-en-charge={}.fly".format(charge),
            options=dft_energy_pcm(gbasis, charge, mult=1, geom=geom)
        )

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method='solution')
        logging.info("Total free energy in solution: {}".format(energy))

        return energy

    G_gas_Hcat                 = -6.28 # kcal/mol; Source: 10.1021/ja010534f, p.7316
    opt_neutral, G_gas_neutral = free_energy_gas(run=False, charge=0)
    opt_anion,   G_gas_anion   = free_energy_gas(run=False, charge=-1)

    G_gas_neutral  *= H_TO_KCAL_MOL
    G_gas_anion    *= H_TO_KCAL_MOL

    print("G_gas_neutral: {}".format(G_gas_neutral))
    print("G_gas_anion: {}".format(G_gas_anion))

    # experimental dG0: 339.14 kcal/mol; Source: https://webbook.nist.gov/cgi/cbook.cgi?ID=C64186&Units=SI&Mask=8#Thermo-React 
    dG0 = G_gas_anion + G_gas_Hcat - G_gas_neutral
    print("dG0_gas: {} KCAL/MOL".format(dG0))

    G_solv_neutral = free_energy_solv(opt_neutral, run=False, charge=0)
    G_solv_anion   = free_energy_solv(opt_anion,   run=False, charge=-1)

    G_solv_neutral *= H_TO_KCAL_MOL
    G_solv_anion   *= H_TO_KCAL_MOL

    print("G_solv_neutral: {}".format(G_solv_neutral))
    print("G_solv_anion:   {}".format(G_solv_anion))

    dG_neutral = G_solv_neutral - G_gas_neutral
    dG_anion   = G_solv_anion - G_gas_anion
    dG_Hcat = -264.61 # kcal/mol; Source: 10.1021/ja010534f, p.7316  
    print("dGs(AH): {} KCAL/MOL".format(dG_neutral))
    print("dGs(A-): {} KCAL/MOL".format(dG_anion))

    ddG = dG_Hcat + dG_anion - dG_neutral
    print("ddG: {} KCAL/MOL".format(ddG))

    dGaq = dG0 + ddG + 1.89
    pK = dGaq / 1.3644
    print("dG0_gas: {} KCAL/MOL".format(dG0))
    print("dGaq: {} KCAL/MOL".format(dGaq))
    print("pK: {}".format(pK))

    ###

    print('------')
    G_solv_Hcat = G_gas_Hcat + dG_Hcat
    dG_direct = G_solv_anion + G_solv_Hcat - G_solv_neutral
    print("[DIRECT] dGaq: {} KCAL/MOL".format(dG_direct))
    print('------')

    #####
    dG_neutral = -7.0
    dG_anion   = -78.0
    dG0        = 338.3

    dG_aq = dG0 + dG_anion + dG_Hcat - dG_neutral + 1.89
    print("[literature] dGaq: {} KCAL/MOL".format(dG_aq))
    pK = dG_aq / 1.3644
    print("[literature] pK: {}".format(pK))



def phenol():
    def free_energy_gas(run=True, charge=0, mult=1):
        gbasis = "ACC-PVDZ"
        logging.info(f" --- PHENOL DFT ENERGY USING BASIS={gbasis} charge={charge} --- ")

        #### INITIAL GEOMETRY FROM CHEMCRAFT
        #geom = [
        #    Atom(symbol="C", charge=6.0, x=3.077618713, y= 0.000000000, z=-1.893164308),
        #    Atom(symbol="C", charge=6.0, x=1.921550020, y=-0.354690359, z=-1.195000308),
        #    Atom(symbol="C", charge=6.0, x=4.233687406, y= 0.354690359, z=-1.195000308),
        #    Atom(symbol="H", charge=1.0, x=1.022124914, y=-0.630640582, z=-1.738173308),
        #    Atom(symbol="H", charge=1.0, x=5.133112512, y= 0.630640582, z=-1.738173308),
        #    Atom(symbol="C", charge=6.0, x=1.921550020, y=-0.354690359, z= 0.201329692),
        #    Atom(symbol="C", charge=6.0, x=4.233687406, y= 0.354690359, z= 0.201329692),
        #    Atom(symbol="H", charge=1.0, x=5.133112512, y= 0.630640582, z= 0.744502692),
        #    Atom(symbol="C", charge=6.0, x=3.077618713, y= 0.000000000, z= 0.899493692),
        #    Atom(symbol="H", charge=1.0, x=3.077618713, y= 0.000000000, z= 1.985840692),
        #    Atom(symbol="H", charge=1.0, x=3.077618713, y= 0.000000000, z=-2.979511308),
        #    Atom(symbol="O", charge=8.0, x=1.136591214, y=-0.832127798, z= 0.925523141),
        #    Atom(symbol="H", charge=1.0, x=1.167760370, y=-0.074590798, z= 2.030895686),
        #]

        # OPTIMIZED GEOMETRY, NEUTRAL PHENOL
        geom = [
            Atom(symbol="C", charge=6.0, x=3.0874765509, y= -0.0101365583, z= -1.8711320889),
            Atom(symbol="C", charge=6.0, x=1.9418949289, y= -0.3291555015, z= -1.1402020000),
            Atom(symbol="C", charge=6.0, x=4.2742081570, y=  0.3492036802, z= -1.2210803661),
            Atom(symbol="H", charge=1.0, x=1.0128936692, y= -0.6097163450, z= -1.6348087086),
            Atom(symbol="H", charge=1.0, x=5.1645981155, y=  0.5977614354, z= -1.7971793838),
            Atom(symbol="C", charge=6.0, x=1.9828636115, y= -0.2884372428, z=  0.2585036530),
            Atom(symbol="C", charge=6.0, x=4.3052622087, y=  0.3853849015, z=  0.1755708406),
            Atom(symbol="H", charge=1.0, x=5.2223451981, y=  0.6625426104, z=  0.6958663949),
            Atom(symbol="C", charge=6.0, x=3.1645231381, y=  0.0677341531, z=  0.9192987400),
            Atom(symbol="H", charge=1.0, x=3.1925563006, y=  0.0972045766, z=  2.0105734110),
            Atom(symbol="H", charge=1.0, x=3.0514265920, y= -0.0430690918, z= -2.9603017292),
            Atom(symbol="O", charge=8.0, x=0.8297376855, y= -0.6083082950, z=  0.9319640896),
            Atom(symbol="H", charge=1.0, x=0.9838650700, y= -0.5470863368, z=  1.8828195864),
        ]

        wrapper = Wrapper.generate_input(
            wd="phenol",
            inpfname="phenol-opt-charge={}.fly".format(charge),
            options=dft_optimize(gbasis, charge=charge, mult=mult, geom=geom)
        )

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method='optimize')[-1]
        logging.info("Energy at optimized geometry: {} Hartree".format(energy))

        geometries = wrapper.opt_geometries(natoms=len(geom))

        opt = geometries[-1]
        logging.info("Optimized geometry (ANG):")
        for atom in opt:
            logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")

        if mult != 2:
            wrapper = Wrapper.generate_input(
                wd="phenol",
                inpfname="phenol-hess-charge={}.fly".format(charge),
                options=dft_hessian(gbasis, charge=charge, mult=mult, geom=opt)
            )

            if run:
                wrapper.clean_wd()
                wrapper.run(link_basis=gbasis)
                wrapper.clean_up()

            wrapper.load_out()
            freqs = wrapper.frequencies()
            zpe = wrapper.parse_zpe()
            thermo = wrapper.thermo()

            positive = all(f >= 0.0 for f in freqs)
            logging.info("Assert freqs > 0: {}".format(positive))
            assert positive

            print("ZPE: {} HARTREE/MOLECULE".format(zpe))

            free_energy_corr = thermo[298.15]['G'] / H_TO_KCAL_MOL
            print("Free energy correction: {} HARTREE".format(free_energy_corr))

            free_energy = energy + zpe + free_energy_corr
            print("Free energy: {} HARTREE".format(free_energy))

            logging.info("---------------------------------------------------------\n")
        else:
            free_energy = energy

        return opt, free_energy

    def free_energy_solv(run=False, geom=None, charge=0, mult=1):
        gbasis = "ACC-PVDZ"

        wrapper = Wrapper.generate_input(
            wd="phenol",
            inpfname="phenol-pcm-en-charge={}.fly".format(charge),
            options=dft_energy_pcm(gbasis, charge=charge, mult=mult, geom=geom)
        )

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method='solution')
        logging.info("Total free energy in solution: {}".format(energy))

        return energy

    def gas_phase_attachment():
        E_phenol_neutral = -307.5163862183
        E_phenol_anion   = -307.5015694141

        ZPE_neutral = 0.104200
        ZPE_anion   = 0.102974

        AU_TO_EV = 27.211
        E_neutral = E_phenol_neutral + ZPE_neutral
        E_anion   = E_phenol_anion   + ZPE_anion
        EA = (E_neutral - E_anion) * AU_TO_EV

        logging.info("Gas phase electron attachment energy: {} eV".format(EA))

    def VIE_phenol_gas():
        E_phenol_neutral = -307.5163862183
        E_phenol_cat     = -307.2120048479

        ZPE_neutral = 0.104200
        ZPE_cat   = 0.102974 ### THIS IS NOT TRUE

        AU_TO_EV = 27.211
        E_neutral = E_phenol_neutral + ZPE_neutral
        E_cat     = E_phenol_cat     + ZPE_cat
        return (E_cat - E_neutral) * AU_TO_EV


    def VIE_phenol_solution():
        AU_TO_EV = 27.211

        E_sol_neutral = -307.519516828
        E_sol_cat     = -307.3001277169

        ZPE_neutral = 0.104200
        ZPE_cat = 0.109274 ### THIS IS NOT TRUE

        E_neutral = E_sol_neutral + ZPE_neutral
        E_cat     = E_sol_cat     + ZPE_cat
        return (E_cat - E_neutral) * AU_TO_EV

    opt_neutral, G_neutral_gas = free_energy_gas(run=False, charge=0, mult=1)
    opt_cation, G_cation_gas = free_energy_gas(run=False, charge=+1, mult=2)

    G_neutral_solv = free_energy_solv(run=False, geom=opt_neutral, charge=0, mult=1)
    G_cation_solv  = free_energy_solv(run=False, geom=opt_cation, charge=+1, mult=2)

    IE_gas = VIE_phenol_gas()
    IE_sol = VIE_phenol_solution()

    IE_gas_exp = 8.51 # eV
    IE_sol_exp = 7.8  # eV 

    logging.info("-------------------------------------")
    logging.info("(Gas phase) calculated   ionization potential: {:.2f} eV".format(IE_gas))
    logging.info("(Gas phase) experimental ionization potential: {:.2f} eV [adiabatic]".format(IE_gas_exp))
    logging.info("(Solution)  calculated   ionization potential: {:.2f} eV".format(IE_sol))
    logging.info("(Solution)  experimental ionization potential: {:.2f} eV".format(IE_sol_exp))
    logging.info("-------------------------------------")

 # XCPCM = 0.5, i.e. COSMO
 # TOTAL FREE ENERGY IN SOLVENT                        =     -307.5243606139 A.U.

# IREP/IDP, only s-functions in dispersion basis set with 1/3 exponent from the aug-cc-pVDZ basis set
#
#  FREE ENERGY IN SOLVENT = <PSI| H(0)+V/2 |PSI>       =     -307.5463315373 A.U.
#  INTERNAL ENERGY IN SOLVENT = <PSI| H(0) |PSI>       =     -307.5147319841 A.U.
#  DELTA INTERNAL ENERGY =  <D-PSI| H(0) |D-PSI>       =        0.0016542342 A.U.
#  ELECTROSTATIC INTERACTION = 1/2(PB+PC) + 1/2PX + UNZ=       -0.0091981606 A.U.
#  PIEROTTI CAVITATION ENERGY                          =        0.0257815840 A.U.
#  DISPERSION FREE ENERGY                              =       -0.0301763157 A.U.
#  REPULSION FREE ENERGY                               =        0.0077749230 A.U.
#  TOTAL INTERACTION (DELTA + ES + CAV + DISP + REP)   =       -0.0058179693 A.U.
#  TOTAL FREE ENERGY IN SOLVENT                        =     -307.5205499533 A.U.
#      PB  =   428.67554897 A.U.   PC  =   428.84842627 A.U.
#      PX  =  -429.08498446 A.U.   UNZ =  -214.22869355 A.U.

# NO $DISBS GROUP => DISPERSION IS CALCULATED USING DEFAULT BASIS: AUG-CC-PVDZ
#
# FREE ENERGY IN SOLVENT = <PSI| H(0)+V/2 |PSI>       =     -307.5452984119 A.U.
# INTERNAL ENERGY IN SOLVENT = <PSI| H(0) |PSI>       =     -307.5149125280 A.U.
# DELTA INTERNAL ENERGY =  <D-PSI| H(0) |D-PSI>       =        0.0014736903 A.U.
# ELECTROSTATIC INTERACTION = 1/2(PB+PC) + 1/2PX + UNZ=       -0.0092106250 A.U.
# PIEROTTI CAVITATION ENERGY                          =        0.0257815840 A.U.
# DISPERSION FREE ENERGY                              =       -0.0289250728 A.U.
# REPULSION FREE ENERGY                               =        0.0077498138 A.U.
# TOTAL INTERACTION (DELTA + ES + CAV + DISP + REP)   =       -0.0046042999 A.U.
# TOTAL FREE ENERGY IN SOLVENT                        =     -307.5195168280 A.U.
#     PB  =   428.67389422 A.U.   PC  =   428.84599638 A.U.
#     PX  =  -429.08092475 A.U.   UNZ =  -214.22869355 A.U.



def phenolate():
    def free_energy_gas(run=True, charge=-1, mult=1):
        gbasis = "ACC-PVDZ"
        logging.info(f" --- PHENOLATE DFT ENERGY USING BASIS={gbasis} charge={charge} --- ")

        # OPTIMIZED GEOMETRY, PHENOLATE ANION
        geom = [
            Atom(symbol="C", charge= 6.0, x= 3.0907826422, y= -0.0090863665, z= -1.8730878818),
            Atom(symbol="C", charge= 6.0, x= 1.9447341925, y= -0.3283230938, z= -1.1491767377),
            Atom(symbol="C", charge= 6.0, x= 4.2915519175, y=  0.3536499986, z= -1.2343411527),
            Atom(symbol="H", charge= 1.0, x= 1.0229139792, y= -0.6073321773, z= -1.6667557839),
            Atom(symbol="H", charge= 1.0, x= 5.1852831172, y=  0.6029126285, z= -1.8089421666),
            Atom(symbol="C", charge= 6.0, x= 1.9062627265, y= -0.3097533080, z=  0.2999158757),
            Atom(symbol="C", charge= 6.0, x= 4.2976899080, y=  0.3832000167, z=  0.1729262609),
            Atom(symbol="H", charge= 1.0, x= 5.2180006709, y=  0.6614080034, z=  0.6986136184),
            Atom(symbol="C", charge= 6.0, x= 3.1631124093, y=  0.0676032391, z=  0.9161200922),
            Atom(symbol="H", charge= 1.0, x= 3.1912527858, y=  0.0972356676, z=  2.0087094936),
            Atom(symbol="H", charge= 1.0, x= 3.0553402767, y= -0.0416672973, z= -2.9677937325),
            Atom(symbol="O", charge= 8.0, x= 0.8628615302, y= -0.5988389882, z=  0.9708849667),
        ]

        wrapper = Wrapper.generate_input(
            wd="phenolate",
            inpfname="phenolate-opt-charge={}.fly".format(charge),
            options=dft_optimize(gbasis, charge=charge, mult=mult, geom=geom)
        )

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method='optimize')[-1]
        logging.info("Energy at optimized geometry: {} Hartree".format(energy))

        geometries = wrapper.opt_geometries(natoms=len(geom))
        opt = geometries[-1]
        logging.info("Optimized geometry (ANG):")
        for atom in opt:
            logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")

        return opt, energy

    def free_energy_solv(run=False, geom=None, charge=0, mult=1):
        gbasis = "ACC-PVDZ"

        wrapper = Wrapper.generate_input(
            wd="phenolate",
            inpfname="phenolate-pcm-en-charge={}.fly".format(charge),
            options=dft_energy_pcm(gbasis, charge=charge, mult=mult, geom=geom)
        )

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method='solution')
        logging.info("Total free energy in solution: {}".format(energy))

        return energy

    def VIE_phenolate_gas():
        E_phenolate     = -306.9518000619
        E_phenolate_rad = -306.8380905507

        AU_TO_EV = 27.211
        return (E_phenolate_rad - E_phenolate) * AU_TO_EV


    def VIE_phenolate_solution():
        E_sol_phenolate     = -307.0430584917
        E_sol_phenolate_rad = -306.842265044

        AU_TO_EV = 27.211
        return (E_sol_phenolate_rad - E_sol_phenolate) * AU_TO_EV

    opt_anion, G_anion_gas = free_energy_gas(run=False, charge=-1, mult=1)
    opt_rad, G_rad_gas     = free_energy_gas(run=False, charge=0, mult=2)

    G_anion_solv = free_energy_solv(run=False, geom=opt_anion, charge=-1, mult=1)
    G_rad_solv   = free_energy_solv(run=False,  geom=opt_rad, charge=0, mult=2)

    IE_gas = VIE_phenolate_gas()
    IE_sol = VIE_phenolate_solution()

    IE_gas_exp = 2.25 # eV
    IE_sol_exp = 7.1  # eV

    logging.info("-------------------------------------")
    logging.info("(Gas phase) calculated   ionization potential: {:.2f} eV".format(IE_gas))
    logging.info("(Gas phase) experimental ionization potential: {:.2f} eV [adiabatic]".format(IE_gas_exp))
    logging.info("(Solution)  calculated   ionization potential: {:.2f} eV".format(IE_sol))
    logging.info("(Solution)  experimental ionization potential: {:.2f} eV".format(IE_sol_exp))
    logging.info("-------------------------------------")

def phenolate_1water():
    def free_energy_gas(run=True, charge=-1, mult=1):
        gbasis = "ACC-PVDZ"
        logging.info(f" --- PHENOLATE DFT ENERGY USING BASIS={gbasis} charge={charge} --- ")

        # OPTIMIZED GEOMETRY, PHENOLATE RADICAL + WATER
        geom = [
            Atom(symbol="C", charge=6.0, x=  3.2647017670, y=  0.0463657916, z= -1.8675326894), 
            Atom(symbol="C", charge=6.0, x=  2.1360814093, y= -0.3007037204, z= -1.1274481271),
            Atom(symbol="C", charge=6.0, x=  4.4528099538, y=  0.4662112463, z= -1.2448613812),
            Atom(symbol="H", charge=1.0, x=  1.2218523497, y= -0.6210664214, z= -1.6310059227),
            Atom(symbol="H", charge=1.0, x=  5.3315938273, y=  0.7371284851, z= -1.8321002423),
            Atom(symbol="C", charge=6.0, x=  2.1218407694, y= -0.2502609309, z=  0.3109867395),
            Atom(symbol="C", charge=6.0, x=  4.4712968680, y=  0.5271806411, z=  0.1591571354),
            Atom(symbol="H", charge=1.0, x=  5.3830454661, y=  0.8513106299, z=  0.6712391904),
            Atom(symbol="C", charge=6.0, x=  3.3528981177, y=  0.1839890412, z=  0.9162673192),
            Atom(symbol="H", charge=1.0, x=  3.3823584383, y=  0.2382423591, z=  2.0070096365),
            Atom(symbol="H", charge=1.0, x=  3.2213276062, y= -0.0094919654, z= -2.9599310402),
            Atom(symbol="O", charge=8.0, x=  1.0909750759, y= -0.5671231870, z=  1.0169593764),
            Atom(symbol="O", charge=8.0, x= -1.1380652064, y= -1.5634562797, z= -0.0040470331),
            Atom(symbol="H", charge=1.0, x= -0.2933156817, y= -1.1403827689, z=  0.3561584472),
            Atom(symbol="H", charge=1.0, x= -0.9279380537, y= -2.5035422239, z=  0.0022223760),
        ]

        wrapper = Wrapper.generate_input(
            wd="phenolate-1water",
            inpfname="phenolate-opt-charge={}.fly".format(charge),
            options=dft_optimize(gbasis, charge=charge, mult=mult, geom=geom)
        )

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method='optimize')[-1]
        logging.info("Energy at optimized geometry: {} Hartree".format(energy))

        geometries = wrapper.opt_geometries(natoms=len(geom))
        opt = geometries[-1]
        logging.info("Optimized geometry (ANG):")
        for atom in opt:
            logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")

        return opt, energy

    def free_energy_solv(run=False, geom=None, charge=0, mult=1):
        gbasis = "ACC-PVDZ"

        wrapper = Wrapper.generate_input(
            wd="phenolate-1water",
            inpfname="phenolate-pcm-en-charge={}.fly".format(charge),
            options=dft_energy_pcm(gbasis, charge=charge, mult=mult, geom=geom)
        )

        if run:
            wrapper.clean_wd()
            wrapper.run(link_basis=gbasis)
            wrapper.clean_up()

        wrapper.load_out()
        energy = wrapper.energy(method='solution')
        logging.info("Total free energy in solution: {}".format(energy))

        return energy

    def VIE_phenolate_solution():
        E_sol_phenolate     = -383.5079865587
        E_sol_phenolate_rad = -383.293209034

        AU_TO_EV = 27.211
        return (E_sol_phenolate_rad - E_sol_phenolate) * AU_TO_EV

    opt_anion, G_anion_gas = free_energy_gas(run=False, charge=-1, mult=1)
    opt_rad, G_rad_gas     = free_energy_gas(run=False, charge=0,  mult=2)

    G_anion_solv = free_energy_solv(run=False, geom=opt_anion, charge=-1, mult=1)
    G_rad_solv   = free_energy_solv(run=False,  geom=opt_rad,  charge=0,  mult=2)

    IE_sol = VIE_phenolate_solution()
    IE_sol_exp = 7.1  # eV

    logging.info("-------------------------------------")
    logging.info("(Solution)  calculated   ionization potential: {:.2f} eV".format(IE_sol))
    logging.info("(Solution)  experimental ionization potential: {:.2f} eV".format(IE_sol_exp))
    logging.info("-------------------------------------")

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --------- PRAK 1 --------------
    #run_cp_rot()
    #run_cp_vib()

    #basis_extrapolation()

    #run_example_01()
    #run_example_02()
    #run_example_03()
    #run_example_04()

    # --------- PRAK 2 --------------
    nitrobenzene()
    #acetic_acid()
    #formic_acid()

    #phenol()
    #phenolate()
    #phenolate_1water()

