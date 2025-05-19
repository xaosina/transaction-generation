import pandas as pd
import yaml
import os

def load_base_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def str_to_df(s):
    s = s.strip().split("\n")
    df = pd.DataFrame(columns=s[0].split("\t"))
    for i, line in enumerate(s[1:]):
        line = line.split("\t")
        df.loc[i] = line
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            try:
                df.iloc[i, j] = eval(df.iloc[i, j])
            except Exception:
                pass
    return df

def df_to_str(df):
    s = "\t".join(df.columns) + "\n"
    for i in range(df.shape[0]):
        values = [str(v) for v in df.iloc[i].tolist()]
        s += "\t".join(values) + "\n"
    return s



def generate_configs(config_rows, common_conf, base_config, output_dir="/home/transaction-generation/zhores/configs"):
    os.makedirs(output_dir, exist_ok=True)

    for _, row in config_rows.iterrows():
        row = row.to_dict()
        row.update(common_conf)
        run_name = row["run_name"]
        config = base_config.copy()

        # Set run name
        config["run_name"] = run_name

        # Update all fields using dot paths
        for key, value in row.items():
            if key == "run_name":
                continue
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value

        # Build filename
        filename = os.path.join(output_dir, f"{run_name}.yaml")
        print("sh transaction-generation/zhores/simple.sh", run_name)
        # Save YAML
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            yaml.dump(config, f, sort_keys=False)

    print(f"✅ Generated {len(config_rows)} config files in '{output_dir}'")