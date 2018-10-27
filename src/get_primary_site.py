import csv
import pandas as pd
import os

# 读取csv至字典
data_dir ="/Users/Tony/Desktop/"
inputfile = data_dir + "miRNA_matrix.csv"
inputmeta = data_dir + "files_meta.tsv"
outputfile = data_dir + "miRNA_matrix_new.csv"
csvFile = open(inputfile, "r")
reader = csv.reader(csvFile)



# 建立空字典
myset = set()
res = 0 
for item in reader:
    # 忽略第一行
    if reader.line_num == 1:
        continue
    myset.add(item[-1])

# example: result["lung"] = num
csvFile.close()

cancer_type = list(myset)
cancer_type_dic = {}

for i, val in enumerate(cancer_type, start = 1):
	cancer_type_dic[val] = i

df = pd.read_csv(inputmeta, sep="\t")
df['classify_label'] = 0
df.loc[df['cases.0.samples.0.sample_type'].str.contains("Normal"), 'classify_label'] = 0
df.loc[df['cases.0.samples.0.sample_type'].str.contains("Tumor"), 'classify_label'] = cancer_type_dic[str(df['cases.0.project.primary_site'])]
df.loc[df['cases.0.samples.0.sample_type'].str.contains("Metastatic"), 'classify_label'] = cancer_type_dic[str(df['cases.0.project.primary_site'])]
df.loc[df['cases.0.samples.0.sample_type'].str.contains("Cancer"), 'classify_label'] = cancer_type_dic[str(df['cases.0.project.primary_site'])]

cancer_count = df.loc[df.classify_label != 0].shape[0]
normal_count = df.loc[df.classify_label == 0].shape[0]

print("{} Normal cases, {} Cancer cases" .format(normal_count, cancer_count))

df.to_csv(outputfile, index = False)
print ("Output Succeed!")
