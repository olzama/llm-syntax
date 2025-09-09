import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def prepare_data_for_plotting(data, human_models, construction_type):
    # Flatten the data into a list of tuples (model, construction_name, frequency)
    records = []
    for model, constructions in list(data[construction_type].items()):
        for constr_name, frequency in list(constructions.items())[:50]:
            # Label the model as either human or LLM based on the model name
            model_type = 'Human' if model in human_models else 'LLM'
            records.append((model, model_type, constr_name, frequency))
    # Convert the list of records to a DataFrame
    df = pd.DataFrame(records, columns=['Model', 'Model Type', 'Construction', 'Frequency'])
    return df

def create_boxplot(df, construction_type, output_filename):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Construction', y='Frequency', hue='Model Type',
                palette="Set2", width=0.8)
    plt.title(f"Boxplot of {construction_type} Frequencies")
    plt.xlabel("Construction Type")
    plt.ylabel("Frequency (Normalized)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_filename)