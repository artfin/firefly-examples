import logging
import os
import re
import shutil
import subprocess

class Wrapper:
    EXE    = "/home/artfin/Desktop/QC-2022/firefly/firefly820"

    def __init__(self, wd=None, inpfile=None):
        self.wd = os.path.abspath(wd)

        if inpfile is None:
            inpfnames = [os.path.join(self.wd, f) for f in os.listdir(self.wd) if f.endswith(".inp")]
            assert len(inpfnames) == 1, "Detected more than one inputfile in the working directory"
            self.inpfname = os.path.abspath(inpfnames[0])

        self.outfname = os.path.splitext(self.inpfname)[0] + '.out'

        logging.info("Initializing wrapper:")
        logging.info("  wd      = {}".format(self.wd))
        logging.info("  inpfname = {}".format(self.inpfname))

    def clean_up(self):
        tmp_dir = self.wd + '.0'
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    def clean_wd(self):
        ignore = lambda f: not (f.endswith(".inp") or f.endswith(".swp"))
        files = [os.path.join(self.wd, f) for f in os.listdir(self.wd) if ignore(f)]

        logging.info("Removing:")
        for f in files:
            logging.info("  {}".format(f))
            os.remove(f)

    def run(self):
        cmd = f"{self.EXE} -r -f -p -stdext -i {self.inpfname} -o {self.outfname} -t {self.wd}"
        proc = subprocess.Popen(cmd, shell=True)
        return proc

    def load_out(self):
        with open(self.outfname, mode='r') as inp:
            self.outfile = inp.readlines()

    def parse_total_energy(self):
        for line in self.outfile:
            if "TOTAL ENERGY" in line:
                word = line.split()[-1]
                if self.is_float(word):
                    return float(word)

    @staticmethod
    def is_float(s):
        pattern = r'^-?\d+(?:\.\d+)$'
        return re.match(pattern, s) is not None

