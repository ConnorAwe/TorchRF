from dig_comm.dataset import create_dataset
from dig_comm.modulations import get_modulation_types
def main():
    kwargs = dict(save_dir='/b0/saustin/saledin/data/digcomm2',
                  modulation_types=get_modulation_types(),
                  num_samples=100,
                  samples_per_symbol=8,
                  iq_len=(256*256*8))
    create_dataset(**kwargs)
    print("done")


if __name__ == "__main__":
    main()
