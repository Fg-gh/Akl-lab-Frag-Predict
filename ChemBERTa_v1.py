import sys
from io import StringIO
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig, get_linear_schedule_with_warmup
import torch.optim as optim
import joblib
from rdkit import Chem
import rdkit.Chem
from rdkit.Chem import AllChem
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torch.utils.data import random_split
from tqdm import tqdm
import warnings
from torch.utils.data.dataloader import default_collate

now = datetime.now()
formatted_time = now.strftime('%y-%m-%d-%H-%M-%S')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

NUM_EPOCHS = 10
INVALID_SMILES_PENALTY = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

warnings.simplefilter("ignore")

#Set this to your local file path before running the code
file_path = "C:\Amish\MLDD\Trial\mTORcanonical.csv"
data_org = pd.read_csv(file_path)

# -- Added by Amish, April 2025
#Code to restrict sample size for quick testing. 
#Comment out the lines below to run the model on the full dataset.
data_org = data_org.sample(n=50, random_state=42)


def write_error_counts_to_file(filename):
    with open(filename, 'w') as file:
        for phase, counts in error_counts.items():
            file.write(f"Error counts for {phase} phase:\n")
            for error_type, count in counts.items():
                file.write(f"  {error_type}: {count}\n")
            file.write("\n")


def clean_smiles(smiles):
    if is_valid_smiles(smiles):
        return smiles
    else:
        # Handle invalid SMILES here (e.g., return a default value, raise an exception, etc.)
        return ""


class SMILESDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        drug_smiles = self.df.iloc[idx]['DRUG SMILES']
        fragment_smiles = self.df.iloc[idx]['FRAG_SMILES']
        if not drug_smiles:  # Skip invalid SMILES strings
            return None
        inputs = self.tokenizer(drug_smiles, max_length=self.max_length, padding='max_length', truncation=True,  return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        fragment_inputs = self.tokenizer(fragment_smiles, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        fragment_ids = fragment_inputs['input_ids'].squeeze(0)
        labels = fragment_ids.clone()

        return {
            **inputs,
            'labels': labels,
            'actual_fragment_smiles': fragment_smiles
        }


tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
config = RobertaConfig.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
model = RobertaForMaskedLM.from_pretrained('seyonec/ChemBERTa-zinc-base-v1', config=config)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return default_collate(batch)


dataset = SMILESDataset(data_org, tokenizer)
size_d = len(dataset)
print("Lenght of the dataset", size_d)
train_size = int(0.7 * size_d)
val_size = int(0.2 * size_d)
test_size = len(dataset) - val_size - train_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                        generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=40, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=40, shuffle=False, collate_fn=collate_fn)

print("length train,val,test dataloaders:", len(train_dataloader), len(val_dataloader), len(test_dataloader))

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * NUM_EPOCHS  # 10 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = CrossEntropyLoss()

losses = []
tanimoto_similarities = []


def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

    return rdkit.Chem.DataStructs.TanimotoSimilarity(fp1, fp2)


def is_valid_smiles_logger(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def is_valid_smiles(smiles, phase):
    old_stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    sys.stderr = old_stderr
    sio.seek(0)
    error_msg = sio.getvalue()
    print("error message:", sio.getvalue())
    phase_errors = error_counts[phase]
    if error_msg:

        if "unclosed ring" in error_msg:
            phase_errors["unclosed ring"] += 1
        elif "duplicated ring closure" in error_msg:
            phase_errors["duplicated ring closure"] += 1
        elif "extra close parentheses" in error_msg:
            phase_errors["extra close parentheses"] += 1
        elif "extra open parentheses" in error_msg:
            phase_errors["extra open parentheses"] += 1
        else:
            print("Not captured ", error_msg)
            phase_errors["other"] += 1
            error_msgs.append(error_msg)
    if mol is not None:
        problems = Chem.DetectChemistryProblems(mol)

        if problems:
            print(problems[0].GetType())
            print(problems[0].Message())
            error_msg = problems[0].Message()
            if "unclosed ring" in error_msg:
                phase_errors["unclosed ring"] += 1
            elif "duplicated ring closure" in error_msg:
                phase_errors["duplicated ring closure"] += 1
            elif "extra close parentheses" in error_msg:
                phase_errors["extra close parentheses"] += 1
            elif "extra open parentheses" in error_msg:
                phase_errors["extra open parentheses"] += 1
            elif "non-ring atom" in error_msg:
                phase_errors["non-ring atom"] += 1
            elif "Can't kekulize" in error_msg:
                phase_errors["can't kekulize"] += 1
            else:
                print("Not captured ", error_msg)
                phase_errors["other"] += 1
                error_msgs.append(error_msg)

    return mol is not None


for param in model.roberta.embeddings.parameters():
    param.requires_grad = False
for layer in model.roberta.encoder.layer[:-4]:
    for param in layer.parameters():
        param.requires_grad = False

model.train()
train_epoch_losses_table = []

train_losses = []
val_losses = []
test_losses = []
train_epoch_losses = []
true_values_epoch = []
predicted_values_epoch = []
t_tanimoto_similarities = []
v_tanimoto_similarities = []

# -- Added by Amish, April 2025
# Initialize the array with different lambda values. 
#Provide only one value if you want the code to run without iteration.
lambda_values = [0.0, 0.000001, 0.00001, 0.0001, 0.001] 

# -- Added by Amish, April 2025
# New Loop over different lambda values for L2 regularization
for lambd in lambda_values:
    
    # Moved inside the lambda loop to reset error tracking for each run.
    # This ensures errors are counted separately for each lambda value.
    # -- Updated by Amish, April 2025
    error_counts = {
        "train": {
            "unclosed ring": 0,
            "duplicated ring closure": 0,
            "extra close parentheses": 0,
            "extra open parentheses": 0,
            "non-ring atom": 0,
            "can't kekulize": 0,
            "other": 0
        },
        "val": {
            "unclosed ring": 0,
            "duplicated ring closure": 0,
            "extra close parentheses": 0,
            "extra open parentheses": 0,
            "non-ring atom": 0,
            "can't kekulize": 0,
            "other": 0
        },
        "test": {
            "unclosed ring": 0,
            "duplicated ring closure": 0,
            "extra close parentheses": 0,
            "extra open parentheses": 0,
            "non-ring atom": 0,
            "can't kekulize": 0,
            "other": 0
        }
    }
    error_msgs = []

    # === Create output folder for this lambda ===
    lambda_str = str(lambd).replace('.', 'p')
    run_folder = f"results_{formatted_time}_lambda_{lambda_str}"
    os.makedirs(run_folder, exist_ok=True)
    print(f"Running training with lambda: {lambd}")
    model = RobertaForMaskedLM.from_pretrained('seyonec/ChemBERTa-zinc-base-v1', config=config)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for layer in model.roberta.encoder.layer[:-4]:
        for param in layer.parameters():
            param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    train_losses = []
    val_losses = []
    t_tanimoto_similarities = []
    v_tanimoto_similarities = []
    train_epoch_losses_table = []

    for epoch in range(NUM_EPOCHS):  # Epoch loop INSIDE lambda loop
        train_batch_losses = []
        for batch in tqdm(train_dataloader, desc=f"Lambda {lambd} - Epoch {epoch + 1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'actual_fragment_smiles'}
            outputs = model(**inputs)
            train_loss = outputs.loss
            predictions = outputs.logits.argmax(dim=-1)
            # Loop through all items in the batch to calculate Tanimoto similarity and validity.
            # Previously only the first item was used, which skipped most of the batch.
            # -- Added by Amish, April 2025
            for i in range(len(predictions)):
                predicted_smiles = tokenizer.decode(predictions[i], skip_special_tokens=True)
                actual_smiles = batch['actual_fragment_smiles'][i]
                similarity = tanimoto_similarity(actual_smiles, predicted_smiles)
                t_tanimoto_similarities.append(similarity)
                
                if not is_valid_smiles(predicted_smiles, 'train'):
                    train_loss += torch.tensor(INVALID_SMILES_PENALTY, device=device)
            ##Removed by Amish, April 2025
            #predicted_smiles = tokenizer.decode(predictions[0], skip_special_tokens=True)
            #actual_smiles = batch['actual_fragment_smiles'][0]
            #similarity = tanimoto_similarity(actual_smiles, predicted_smiles)
            #t_tanimoto_similarities.append(similarity)
            #if not is_valid_smiles(predicted_smiles, 'train'):
            #    train_loss += torch.tensor(INVALID_SMILES_PENALTY, device=device)
            
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            train_loss += lambd * l2_norm
            train_loss.mean().backward()
            optimizer.step()
            scheduler.step()
            train_batch_losses.append(train_loss.mean().item())
        print("Trains_batch_losses = ", train_batch_losses)
        train_epoch_losses.append(train_batch_losses)

        model.eval()
        val_loss_over_batch = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Lambda {lambd} - Validation {epoch + 1}/{NUM_EPOCHS}"):
                inputs = {key: val.to(device) for key, val in batch.items() if key != 'actual_fragment_smiles'}
                outputs = model(**inputs)
                val_loss = outputs.loss
                predictions = outputs.logits.argmax(dim=-1)
                
                # Iterate over each item in the validation batch to compute Tanimoto similarity and check SMILES validity.
                # This ensures the entire batch is evaluated, not just the first entry.
                # -- Added by Amish, April 2025
                for i in range(len(predictions)):
                    predicted_smiles = tokenizer.decode(predictions[i], skip_special_tokens=True)
                    actual_smiles = batch['actual_fragment_smiles'][i]
                    similarity = tanimoto_similarity(actual_smiles, predicted_smiles)
                    v_tanimoto_similarities.append(similarity)

                    if not is_valid_smiles(predicted_smiles, 'val'):
                        val_loss += torch.tensor(INVALID_SMILES_PENALTY, device=device)     
                    
                val_loss_over_batch.append(val_loss.mean().item())      
                
                ##Removed by Amish, April 2025        
                #predicted_smiles = tokenizer.decode(predictions[0], skip_special_tokens=True)
                #actual_smiles = batch['actual_fragment_smiles'][0]
                #similarity = tanimoto_similarity(actual_smiles, predicted_smiles)
                #v_tanimoto_similarities.append(similarity)
                #if not is_valid_smiles(predicted_smiles, 'val'):
                #    val_loss += torch.tensor(INVALID_SMILES_PENALTY, device=device)
                #val_loss_over_batch.append(val_loss.mean().item())

        val_loss = np.mean(val_loss_over_batch)
        print(f"Lambda: {lambd}, Epoch: {epoch + 1}, Train Loss: {np.mean(train_batch_losses)}, Val Loss: {val_loss}")
        train_epoch_losses_table.append((epoch + 1, np.mean(train_batch_losses), val_loss))
        train_losses.append(train_batch_losses)
        val_losses.append(val_loss_over_batch)
        
    model.eval()
    true_values = []
    predicted_values = []
    loss_values = []
    test_losses = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            test_loss = 0.0
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'actual_fragment_smiles'}
            outputs = model(**inputs)
            loss = outputs.loss
            test_loss += loss.mean().item()
            predictions = outputs.logits.argmax(dim=-1)
            
            # Decode the predicted SMILES and the actual SMILES
            # Process all items in the test batch to collect SMILES predictions, similarities, and test loss.
            # Fixes the bug where only the first prediction in the batch was being used.
            # -- Added by Amish, April 2025
            for i in range(len(predictions)):
                predicted_smiles = tokenizer.decode(predictions[i], skip_special_tokens=True)
                actual_smiles = batch['actual_fragment_smiles'][i]

                if not is_valid_smiles(predicted_smiles, 'test'):
                    test_loss += torch.tensor(INVALID_SMILES_PENALTY, device=device)

                test_losses.append(test_loss)
                true_values.append(actual_smiles)
                predicted_values.append(predicted_smiles)
                loss_values.append(test_loss)
            
            ##Removed by Amish, April 2025
            #predicted_smiles = tokenizer.decode(predictions[0], skip_special_tokens=True)
            #if not is_valid_smiles(predicted_smiles, 'test'):
            #    test_loss += torch.tensor(INVALID_SMILES_PENALTY, device=device)
            #actual_smiles = batch['actual_fragment_smiles'][0]
            
            #test_losses.append(test_loss)
            #true_values.append(actual_smiles)
            #predicted_values.append(predicted_smiles)
            #loss_values.append(test_loss)

    test_losses = [float(x) if isinstance(x, torch.Tensor) else x for x in test_losses]
    print("Test losses", test_losses)
    epoch_losses_df = pd.DataFrame(train_epoch_losses_table, columns=['epoch', 'train_loss', 'val_loss'])
    epoch_losses_df.to_csv(os.path.join(run_folder, "epoch_losses_with_val.csv"), index=False)

    test_losses_df = pd.DataFrame(list(zip(true_values, predicted_values, test_losses)),
                                  columns=['true_smile', 'predicted_smile', 'loss'])
    test_losses_df.to_csv(os.path.join(run_folder, "test_losses_with_val.csv"), index=False)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(os.path.join(run_folder, "model"))
    tokenizer.save_pretrained(os.path.join(run_folder, "tokenizer"))
    joblib.dump(config, os.path.join(run_folder, "config.pkl"))

    true_values = []
    predicted_values = []
    model.eval()
    batch_index = []

    train_mean_losses = [np.mean(epoch_losses) for epoch_losses in train_losses]
    val_mean_losses = [np.mean(epoch_losses) for epoch_losses in val_losses]
    train_epoch_mean_similarities = []
    l = len(t_tanimoto_similarities)
    chunk_size = int(l / NUM_EPOCHS)

    for i in range(NUM_EPOCHS):
        epoch_losses = []
        start_idx = i * len(train_dataloader)
        end_idx = start_idx + len(train_dataloader)
        print("Start, end indx", start_idx, end_idx)
        train_mean_similarity = np.mean(t_tanimoto_similarities[start_idx:end_idx])
        train_epoch_mean_similarities.append(train_mean_similarity)

    val_epoch_mean_similarities = []
    for i in range(NUM_EPOCHS):
        epoch_losses = []
        start_idx = i * len(val_dataloader)
        end_idx = start_idx + len(val_dataloader)

        val_mean_similarity = np.mean(v_tanimoto_similarities[start_idx:end_idx])
        val_epoch_mean_similarities.append(val_mean_similarity)
    print("length train,val,test dataloaders:", len(train_dataloader), len(val_dataloader), len(test_dataloader),
          len(t_tanimoto_similarities))
    print("val, train epoch mean similarities", train_epoch_mean_similarities, val_epoch_mean_similarities)
    plt.figure(figsize=(14, 6))

    # Plot for Training Loss per Epoch
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), train_mean_losses, marker='o', color='b', label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_mean_losses, marker='x', color='r', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xticks(range(1, NUM_EPOCHS + 1))
    plt.grid(True)
    plt.legend()

    # Plot for Mean Tanimoto Similarity per Epoch
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), train_epoch_mean_similarities, marker='o', color='b',
             label='Train Mean Tanimoto Similarity')
    plt.plot(range(1, NUM_EPOCHS + 1), val_epoch_mean_similarities, marker='x', color='r',
             label='Val Mean Tanimoto Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Tanimoto Similarity')
    plt.title('Mean Tanimoto Similarity Over Epochs')
    plt.xticks(range(1, NUM_EPOCHS + 1))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "mTORfigure_train_val.png"))
    # Print the error counts
    print("Error message counts:")
    for phase, counts in error_counts.items():
        print(f"Error counts for {phase} phase:")
        for error_type, count in counts.items():
            print(f"  {error_type}: {count}")
        print()
    write_error_counts_to_file(os.path.join(run_folder, "error_counts.txt"))