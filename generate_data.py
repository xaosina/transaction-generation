import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer
import argparse
import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--sample", type=float, default=.2)
    parser.add_argument("-g", "--gpu", type=str, default="cuda")
    args = parser.parse_args()

    orig_data = pd.read_csv("datafusion/preprocessed_with_id_train.csv")
    client_ids = orig_data["user_id"].unique()
    n_clients = int(len(client_ids) * args.sample)

    gen = np.random.default_rng(0)
    train_ids = pd.Series(
        gen.choice(client_ids, size=n_clients, replace=False),
        name="user_id",
    )
    dataset = orig_data.merge(train_ids, on="user_id")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dataset)
    metadata.update_columns_metadata(
        {
            "user_id": {"sdtype": "id"},
            "mcc_code": {"sdtype": "categorical"},
        }
    )
    metadata.set_sequence_key("user_id")
    metadata.set_sequence_index("days_since_first_tx")

    metadata.validate()

    synthesizer = PARSynthesizer(
        metadata,
        context_columns=["customer_age", "dummy_binclass"],
        epochs=args.epoch,
        verbose=True,
        cuda=args.gpu
    )

    synthesizer.fit(dataset)
    synthesizer.save("my_synthesizer.pkl")

    synthetic_data = synthesizer.sample(num_sequences=10000)
    synthetic_data.to_csv("tabsyn-concat/synthetic/PAR/synthetic.csv")
