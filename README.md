## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df = pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2024-10-03 105623](https://github.com/user-attachments/assets/4d7a68ac-1ac6-467d-8f32-dcc10252fc6e)

```
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
pm = ['Hot','Warm','Cold']
e1 = OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])
```
![Screenshot 2024-10-03 105710](https://github.com/user-attachments/assets/d6337134-02cb-47e2-8633-215af83b59d7)
```
df['bo2']=e1.fit_transform(df[['ord_2']])
df
```
![Screenshot 2024-10-03 105756](https://github.com/user-attachments/assets/4cd5ae97-764d-4bae-8bae-ec27eb668e7e)

```
le = LabelEncoder()
dfc = df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-10-03 105840](https://github.com/user-attachments/assets/925527b7-26d2-42e5-a97f-704b23a55737)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2

```
![image](https://github.com/user-attachments/assets/e8171fbe-60c6-4564-a3f3-226f2c194464)
```
pd.get_dummies(df2,columns=['nom_0'])
```
![image](https://github.com/user-attachments/assets/9887a78e-a705-490a-9123-bf5d48a96980)
```
pip install --upgrade category_encoders
```
![Screenshot 2024-10-03 110039](https://github.com/user-attachments/assets/c6a244b1-537a-46a2-a18e-d632ba8883b3)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![Screenshot 2024-10-03 110141](https://github.com/user-attachments/assets/a9be848d-c03b-4617-a8e5-14f82312e3d1)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-10-03 110252](https://github.com/user-attachments/assets/db60b99e-6f42-4cff-8a94-60c849532bba)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![Screenshot 2024-10-03 110322](https://github.com/user-attachments/assets/537ea972-1d8a-441d-9ca6-99d253e50197)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![Screenshot 2024-10-03 110349](https://github.com/user-attachments/assets/d6b7a8de-b6c4-49be-aeca-a5167f018a45)
```
df.skew()
```
![Screenshot 2024-10-03 110414](https://github.com/user-attachments/assets/25645f9b-e54d-4272-9982-5287baf26792)
```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-10-03 110442](https://github.com/user-attachments/assets/4dac5619-e880-4c63-b69a-ba9ea3cd9267)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2024-10-03 110510](https://github.com/user-attachments/assets/f94dd54b-b235-42de-a30d-8121e1c0e4f3)
```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2024-10-03 110541](https://github.com/user-attachments/assets/c695444f-6307-497b-919d-fb1827f9074d)
```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2024-10-03 110606](https://github.com/user-attachments/assets/5fb1eb35-f14d-4caf-b5a1-243c300dba99)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-10-03 110631](https://github.com/user-attachments/assets/4e14eeea-8382-4ba6-8144-0f9a78dacdc2)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![Screenshot 2024-10-03 110737](https://github.com/user-attachments/assets/7827a3e2-7775-4dd6-b546-8b924b31bc64)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-03 110809](https://github.com/user-attachments/assets/7e72e256-a3a8-4db3-b1d8-67e4e551495e)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/73dc4f09-732f-49f4-b2cf-ddee7241daa2)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])


sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/6d1d7ad0-58f8-4415-9682-19d4d3ecc33e)
```
df
```
![Screenshot 2024-10-03 110922](https://github.com/user-attachments/assets/102fa98e-d122-45e3-ae2d-a62d7a006c82)
```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c09512b4-0953-43b7-a8ab-61047323055e)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/6179c80e-bf8b-427c-8395-03954ad49051)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9a034180-5af2-4bcd-b76f-553ed83bb299)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/438fb46f-cbdc-4a38-886a-44d820aab6cd)


# RESULT:
Thus the program to read the given data and perform Feature Encoding and Transformation process and save the data to a file is successfully executed

       
