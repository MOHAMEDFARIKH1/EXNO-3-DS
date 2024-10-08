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
df = pd.read_csv("Encoding Data.csv")
df.head()
```
![Screenshot 2024-10-08 095441](https://github.com/user-attachments/assets/874f4dad-96b9-4e73-a08f-5bfafbb7576d)

```
df.tail()
```
![Screenshot 2024-10-08 095444](https://github.com/user-attachments/assets/d0e318a4-1e75-4430-886c-58025d90828b)

```
df.describe()
```
![Screenshot 2024-10-08 095449](https://github.com/user-attachments/assets/08489405-b231-4676-8c71-6248dd7474a5)

```
df.info()
```
![Screenshot 2024-10-08 095453](https://github.com/user-attachments/assets/1e94ce81-880c-44ee-8e51-aa6f9f3abc73)

```
df.shape
```
![Screenshot 2024-10-08 095455](https://github.com/user-attachments/assets/4eac499e-7e1a-47ba-85a5-3eb1af435297)

```
df
```
![Screenshot 2024-10-08 095459](https://github.com/user-attachments/assets/610cedfc-aca9-4786-bcae-9494b81d4520)

```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot', 'Warm','Cold']
oe=OrdinalEncoder(categories=[pm])
oe.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-10-08 095503](https://github.com/user-attachments/assets/9ff2687a-eda6-4bd8-a834-672bb62a6c30)

```
df['bo2']=oe.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-10-08 095507](https://github.com/user-attachments/assets/72824cdb-bf82-4b9c-85fd-f440bc9dd869)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-10-08 095511](https://github.com/user-attachments/assets/7bd86a5e-59cc-438c-9f37-c463730b9597)

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-10-08 095514](https://github.com/user-attachments/assets/ecdb7a9f-6123-45d0-bd43-ac1b174edd89)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2024-10-08 095518](https://github.com/user-attachments/assets/ed874260-e9ad-4b14-be5c-67f32c84c463)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![Screenshot 2024-10-08 095522](https://github.com/user-attachments/assets/ac8ef650-108c-4c17-9a32-b25e6fabaed7)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-10-08 095527](https://github.com/user-attachments/assets/91505b38-dfe3-458a-b9b7-8b4862efb89c)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![Screenshot 2024-10-08 095531](https://github.com/user-attachments/assets/9f62c83e-249d-4523-a1fe-9e6a02599e89)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![Screenshot 2024-10-08 095536](https://github.com/user-attachments/assets/fc9bd46e-4052-47c3-9f75-23de8e6202cf)

```
df.skew()
```
![Screenshot 2024-10-08 095540](https://github.com/user-attachments/assets/6d4e7777-8dc3-404d-9914-d2138e62f8ae)

```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-10-08 095542](https://github.com/user-attachments/assets/1098433c-f991-4494-b8e2-7b17880de97c)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2024-10-08 095545](https://github.com/user-attachments/assets/1df3a58f-e9db-48bc-a2bc-5b83cd7cdb79)

```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2024-10-08 095549](https://github.com/user-attachments/assets/16acb480-21e3-4ad6-8d74-b21726679357)

```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2024-10-08 095553](https://github.com/user-attachments/assets/6129b474-79db-439e-a90d-b8acb29761e5)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-10-08 095559](https://github.com/user-attachments/assets/cea6ab54-b5e0-43bd-a919-6a62e7c79551)

```
df["Moderate Negative Skew_yeojohnson"],parameters =stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![Screenshot 2024-10-08 095606](https://github.com/user-attachments/assets/30ac4269-c83d-429f-afa0-f4c53eed6608)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-08 095612](https://github.com/user-attachments/assets/36aef3d7-35ee-4a95-bcd1-cb93f1c072ec)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-08 095618](https://github.com/user-attachments/assets/fdf41413-1c4a-4115-aa62-6515755a7013)

```
df
```
![Screenshot 2024-10-08 095624](https://github.com/user-attachments/assets/dc855ea1-e033-414a-a483-642a1ff1f684)

```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![Screenshot 2024-10-08 095630](https://github.com/user-attachments/assets/b83dda79-ce7b-4865-8c35-5e058af86675)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![Screenshot 2024-10-08 095636](https://github.com/user-attachments/assets/aad5e0d3-52e1-4af8-8e3f-81902590b1aa)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![Screenshot 2024-10-08 095641](https://github.com/user-attachments/assets/70c3deff-ac73-41b8-8126-d34be8043ffe)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-10-08 095645](https://github.com/user-attachments/assets/9f7513b7-a996-4ae8-90f7-882cbf37cd2f)

# RESULT:
  Thus in the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
