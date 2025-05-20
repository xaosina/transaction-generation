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
        values = []
        for v in df.iloc[i].tolist():
            if isinstance(v, float):
                values += [str(v).replace(".", ",")]
            else:
                values += [str(v)]
        s += "\t".join(values) + "\n"
    return s

def compare_dicts(dict1, dict2):
    differences = {}
    all_keys = set(dict1.keys()).union(set(dict2.keys()))

    for key in all_keys:
        val1 = dict1.get(key, None)
        val2 = dict2.get(key, None)

        if val1 != val2:
            differences[key] = (val1, val2)

    return differences

def save_config_if_changed(filename, config):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                current = yaml.safe_load(f)
            diff = compare_dicts(config, current)
            if diff:
                print(f"WARNING: {filename} exists and content differs. {diff}.")
        except yaml.YAMLError:
            print(f"Warning: {filename} has invalid YAML. Not overwriting.")
        return

    with open(filename, "w") as f:
        yaml.dump(config, f, sort_keys=False)


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
        save_config_if_changed(filename, config)

        with open(filename, "w") as f:
            yaml.dump(config, f, sort_keys=False)
    print(f"✅ Generated {len(config_rows)} config files in '{output_dir}'")
    
    for _, row in config_rows.iterrows():
        run_name = row["run_name"]
        print("sh zhores/simple_mega.sh", run_name)

def collect_res(df, cols = None):
    new_res = df.copy()
    for i, row in new_res.iterrows():
        path = row["run_name"]
        df = pd.read_csv(f"log/generation/{path}/results.csv", index_col=0)
        test_cols = [col for col in df.index if ("test_" in col)]
        if cols is not None:
            test_cols = [col for col in test_cols if (col in cols)]
        new_res.loc[i, test_cols] = df.loc[test_cols, "mean"]
    return new_res