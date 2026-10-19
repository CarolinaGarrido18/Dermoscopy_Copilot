import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from sklearn.model_selection import StratifiedGroupKFold


# Descarregar os dois datasets 

# Imagens + metadata
# path_img = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
# print("Imagens:", path_img)

# # Máscaras de segmentação (referência [12] da proposta)
# path_seg = kagglehub.dataset_download("tschandl/ham10000-lesion-segmentations")
# print("Máscaras:", path_seg)

path_img = "C:\\Users\\carol\\.cache\\kagglehub\\datasets\\kmader\\skin-cancer-mnist-ham10000\\versions\\2"
path_seg = "C:\\Users\\carol\\.cache\\kagglehub\\datasets\\tschandl\\ham10000-lesion-segmentations\\versions\\1"

# Ver o que há nas pastas
for p in [path_img, path_seg]:
    print("\n", p)
    for item in os.listdir(p):
        print("  ", item)

# Carregar e explorar o dataset
df = pd.read_csv(os.path.join(path_img, "HAM10000_metadata.csv"))
print(df.shape)          # deve dar (10015, 7)
print(df.head())
print(df["dx"].value_counts())   # qts imagens existem para cada tipo de diagnóstico (desiquilíbrio)
print(df["lesion_id"].value_counts().value_counts()) # nº de imagens por lesão (qts lesões têm só 1 img, qts têm 2...)

# Ligar cada image_id ao ficheiro da imagem
# Procura todas as imagens .jpg nas pastas e cria uma lista que liga o nome da 
# imagem ao local onde ela está no PC
img_paths = {os.path.splitext(os.path.basename(f))[0]: f
             for f in glob(os.path.join(path_img, "*", "*.jpg"))}
# Cria uma nova coluna na nossa tabela virtual df (chamada "path") e guarda lá o endereço exato de cada imagem
df["path"] = df["image_id"].map(img_paths)
# Verifica se alguma imagem se perdeu
print(df["path"].isna().sum())   # tem de dar 0, encontrou as imagens todas com sucesso

# Ver algumas imagens (confirmar que está tudo certo)
classes = df["dx"].unique() # dx é a coluna dos diagnósticos
fig, axes = plt.subplots(1, len(classes), figsize=(18, 3))
for ax, c in zip(axes, classes): # axes: onde vai ficar a imagem, classes: nome da doença
    row = df[df["dx"] == c].iloc[0]
    ax.imshow(Image.open(row["path"]))
    ax.set_title(c)
    ax.axis("off")
plt.show()

# Dividir as imagens em três grupos diferentes: Treino (para o modelo aprender), Validação 
# (para afinar o modelo enquanto aprende) e Teste (a prova final).

# 1) separar teste (~15%) (divide em 7 fatias, cada fatia ~15%)
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
# Corta os dados. Olha para o diagnóstico ("dx") para manter as proporções 
# e agrupa pelo ID da mancha ("lesion_id") para não separar fotos da mesma lesão
trainval_idx, test_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
# corta a tabela original (df) em duas novas tabelas: a test e a trainval
trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

# 2) separar validação (~15% do total) do que sobrou
sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
# Corta os dados que sobraram da mesma forma (mantendo proporções e agrupando por lesão)
train_idx, val_idx = next(sgkf2.split(trainval, trainval["dx"], groups=trainval["lesion_id"]))
# Cria as tabelas definitivas de Treino e de Validação
train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]

# Verificações:
# Mostra quantas imagens ficaram no Treino, Validação e Teste
print(len(train), len(val), len(test))
# Procura se alguma lesão do treino está no teste
print("Lesões em comum treino/teste:", set(train.lesion_id) & set(test.lesion_id))  # tem de ser vazio (set())
# Procura se alguma lesão do treino está na validação
print("Lesões em comum treino/val:", set(train.lesion_id) & set(val.lesion_id))     # tem de ser vazio (set())
# Mostra quantas doenças de cada tipo foram para o teste
print(test["dx"].value_counts())   # todas as classes devem aparecer

os.makedirs("splits", exist_ok=True)
cols = ["image_id", "lesion_id", "dx"]
train[cols].to_csv("splits/train.csv", index=False)
val[cols].to_csv("splits/val.csv", index=False)
test[cols].to_csv("splits/test.csv", index=False)
print("Splits guardados!")
