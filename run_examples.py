import logging
import time
from wrapper import Wrapper

def run_example_1():
    wrapper = Wrapper(wd="1_co2_rhf_en")
    wrapper.clean_wd()

    proc = wrapper.run()
    while proc.poll() is None:
        time.sleep(0.5)

    wrapper.clean_up()

    wrapper.load_out()
    total_energy = wrapper.parse_total_energy()

    logging.info("Total energy: {}".format(total_energy))



if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    run_example_1()
