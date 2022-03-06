import logging
import time

import logging
import os
import re
import shutil
import subprocess
from collections import namedtuple

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

        logging.info("Initializing wrapper:")
        logging.info("  wd      = {}".format(self.wd))
        logging.info("  inpfname = {}".format(self.inpfname))

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

    def energy(self):
        for line in self.outfile:
            if "TOTAL ENERGY" in line:
                word = line.split()[-1]
                if self.is_float(word):
                    return float(word)

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

    @staticmethod
    def is_float(s):
        pattern = r'^-?\d+(?:\.\d+)$'
        return re.match(pattern, s) is not None

def run_example_01():
    logging.info(" --- CO2 RHF ENERGY CALCULATION USING BASIS=STO-3G --- ")
    wrapper = Wrapper(wd="1_co2_rhf_en", inpfname="1_co2_rhf_en-basis=sto-3g.fly")
    wrapper.clean_wd()
    wrapper.run()
    wrapper.clean_up()
    wrapper.load_out()
    energy = wrapper.energy()
    logging.info("Total energy: {}".format(energy))
    logging.info("---------------------------------------------------------\n")

    logging.info(" --- CO2 RHF ENERGY CALCULATION USING BASIS=CC-PVDZ --- ")
    wrapper = Wrapper(wd="1_co2_rhf_en", inpfname="1_co2_rhf_en-basis=cc-pvdz.fly")
    wrapper.clean_wd()
    wrapper.run(link_basis="cc-pvdz")
    wrapper.clean_up()
    wrapper.load_out()
    energy = wrapper.energy()
    logging.info("Total energy: {}".format(energy))
    logging.info("---------------------------------------------------------\n")

def run_example_02():
    wrapper = Wrapper(wd="2_co2_rhf_opt", inpfname="2_co2_rhf_opt.fly")
    wrapper.clean_wd()

    proc = wrapper.run()
    wrapper.load_out()
    geometries = wrapper.opt_geometries()

    opt = geometries[-1]
    logging.info("Optimized geometry (ANG):")
    for atom in opt:
        logging.info(f"  {atom.symbol} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")

    logging.info("Optimized geometry (BOHR):")
    for atom in opt:
        logging.info(f"  {atom.symbol} {atom.x / BOHR_TO_ANG:.10f} {atom.y / BOHR_TO_ANG:.10f} {atom.z / BOHR_TO_ANG:.10f}")

    wrapper.clean_up()

    wrapper = Wrapper(wd="2_co2_rhf_opt", inpfname="2_co2_rhf_hess.fly")
    proc = wrapper.run()
    wrapper.load_out()
    freqs = wrapper.frequencies()

    logging.info("Frequencies at optimized geometry (cm-1):")
    for f in freqs:
        logging.info("  {:.3f}".format(f))

    p = all(f > 0.0 for f in freqs)
    logging.info("Assert freqs > 0: {}".format(p))
    assert p

    wrapper.clean_up()

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    run_example_01()
    #run_example_02()
