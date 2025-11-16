import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle
import os

print("Lets see the Dataset Values.........\n")
try:
    df = pd.read_csv("Data Sheet/laptops.csv")
except FileNotFoundError:
    print("File not found")
except Exception as e:
    print(f"Error found : {e}") 


print("Lets see some data of this data sheet..........\n")
df.head()

print("Lets See the Shape of the Datasets............\n")
print(f"The Total Columns of the Data set is {df.shape[1]} and the total rows of the Dataset is {df.shape[0]}.\n")

print("Let see there is any Null Values or not..........\n")
print(df.isnull().sum())

print("Lets see the Data Types of the Dataset Columns.\n")
print(df.dtypes)

print("Lets Describe all the Dataset Values....................\n")
df.describe(include='all')

print("Lets remove the null values of this datasets........")

df['Storage type'] = df['Storage type'].fillna(df['Storage type'].mode()[0])
df['GPU'] = df['GPU'].fillna(df['GPU'].mode()[0])
df['Screen'] = df['Screen'].fillna(df['Screen'].mean())  

print("Null Value Removed Successfully..........")

print("Lets check still there is any null values present or not...............\n ")
print(df.isnull().sum())

print("Lets Check there is any outliers or not............\n")

temp = ['RAM','Storage','Screen']
for i in temp:
    plt.figure(figsize=(8,4))
    sns.boxplot(x = df[i])
    plt.title(f"{i} Outliers Checking")
    plt.tight_layout()
    plt.show()
print("There is no major OutLiers present.........\n")

print("Lets Arrange the Columns of the Datasets..............\n")
df.drop('Laptop',axis=1,inplace=True)
print("Removed 'Laptop' Column from this dataset.....................\n")

df.head()

#Lets Rearrange the Columns Name
print("Lets Rearange the Columns..............\n")
rearrange_columns = ['Brand','Model','CPU','Status','RAM','Storage','Storage type','GPU','Screen','Touch','Final Price']
df = df.reindex(columns=rearrange_columns)
print(df.head())

print("Distribution of Final Price.")

plt.figure(figsize=(8,5))
sns.histplot(df['Final Price'], bins=40,color="red",edgecolor="black")
plt.title("Distribution of Laptop Final Price")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("Diagram Images/Distribution of Final Price.png",dpi=200,)
plt.show()


#RAM vs Final Price

plt.figure(figsize=(8,4))
plt.scatter(df['RAM'],df['Final Price'],color="green",edgecolor="black")
plt.title("RAM vs Final Price")
plt.xlabel("RAM (GB)")
plt.ylabel("Price (in $)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Diagram images/RAM_vs_Price.png",dpi=200,bbox_inches='tight')
plt.show()

branch_unique = df['Brand'].unique()
Model_uniqie=df['Model'].unique()
CPU_unique = df['CPU'].unique()
GPU_unique = df['GPU'].unique()

print(branch_unique)
print(Model_uniqie)
print(CPU_unique)
print(GPU_unique)

df.dtypes

#Lets use the Encodeing Technique for the Labeled Columns.

print("We are Encode the Labeled Dataset using the Label Encoder Functions............\n")

temp1 = ['Brand','Model','CPU','Status','Storage type','GPU','Touch']

encoders = {}   

for col in temp1:
    lb = LabelEncoder()
    df[col] = lb.fit_transform(df[col])
    encoders[col] = lb 

print("Encoding Completed Using Label Encoder................\n")
print(df.head())

encoders

#Define The X And Y Columns for the Training and testing data.
print("Define The X And Y Columns for the Training and testing data..........\n")
x = df.iloc[:,:-1]
y = df['Final Price']

#Train Test Splot
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Lets Rescal the Dataset
print("Lets Rescale the Datasets using Standard Scaler.......\n")

scl = StandardScaler()
x_train_scaled = scl.fit_transform(x_train)
x_test_scaled = scl.transform(x_test)

#I am Use the Linear Regression:- 
print("Now We are use the Linear Regression Model for Price Prediction..........\n")

lr = LinearRegression()

lr.fit(x_train_scaled, y_train)
y_predict_lr = lr.predict(x_test_scaled)

print("Mean Absolute Error(MAE):",round(mean_absolute_error(y_test,y_predict_lr),2))
print("Mean Squared Error(MSE):",round(mean_squared_error(y_test,y_predict_lr),2))
print(f"R2 Score(Model Accuracy) is ",round(r2_score(y_test,y_predict_lr),2)*100,"%")

#Lets Try The KNeighbors Regressor Model and Check the Accuracy.

knn = KNeighborsRegressor(n_neighbors=4)

knn.fit(x_train_scaled,y_train)
y_predict_knn = knn.predict(x_test_scaled)

print("Mean Absolute Error(MAE):",round(mean_absolute_error(y_test,y_predict_knn),2))
print("Mean Squared Error(MSE):",round(mean_squared_error(y_test,y_predict_knn),2))
print(f"R2 Score(Model Accuracy) is ",round(r2_score(y_test,y_predict_knn),2)*100,"%")

#Lets Check the Random Forest Regressor model
random_forest = RandomForestRegressor(random_state=42,n_estimators=200,max_depth=None)

random_forest.fit(x_train_scaled,y_train)
y_predict_rfr=random_forest.predict(x_test_scaled)

print("Mean Absolute Error(MAE):",round(mean_absolute_error(y_test,y_predict_rfr),2))
print("Mean Squared Error(MSE):",round(mean_squared_error(y_test,y_predict_rfr),2))
print(f"R2 Score(Model Accuracy) is ",round(r2_score(y_test,y_predict_rfr),2)*100,"%")

#Lets Use the XgBoost Regressor

xgb = XGBRegressor(random_state=42,
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=9)

xgb.fit(x_train_scaled,y_train)
y_predict_xgb = xgb.predict(x_test_scaled)

print("Mean Absolute Error(MAE):",round(mean_absolute_error(y_test,y_predict_xgb),2))
print("Mean Squared Error(MSE):",round(mean_squared_error(y_test,y_predict_xgb),2))
print(f"R2 Score(Model Accuracy) is ",round(r2_score(y_test,y_predict_xgb),2)*100,"%")

df['Storage type'].unique()

#User Input Module
print("Enter the User Input........\n")


Brand = input("Enter Brand of Laptop(Company Name):")
Model = input("Enter Model of Laptop(Like:Notebook,inspioron 15):")
CPU = input("Enter CPU of Laptop(Like:AMD Ryzen 7,intel core 8):")
Status = input("Enter the Status(ex.New or refurbished):")
RAM = int(input("Enter the RAM(ex. 8):"))
Storage = int(input("Enter the Storage(ex. 512):"))
Storage_type = input("Enter the Storage type of the Laptop(ex. HDD,SSD):")
GPU = input("Enter GPU of Laptop(Like:RTX 3050,RTX 3090):")
Screen = float(input("Enter the Screen Size into Inches(Ex. 15.6):"))
Touch = input("Enter the Touch Screen Yes/No :")

new_df = {
    "Brand":Brand,
    "Model":Model,       
    "CPU":CPU,       
    "Status":Status,         
    "RAM":RAM,           
    "Storage":Storage,       
    "Storage type":Storage_type, 
    "GPU":GPU,
    "Screen":Screen,       
    "Touch":Touch      
}
 
new_df = pd.DataFrame([new_df])

#Encode all the Catagorical data Values.
for i in temp1:
    if new_df[i][0] in encoders[i].classes_:
        new_df[i] = encoders[i].transform(new_df[i])
    else:
        new_df[i] = [-1]

new_df = new_df[x.columns]

#scal down all the userinput data.

scaled_data = scl.transform(new_df)

lr_predicted_price = lr.predict(scaled_data)
rfr_predicted_price = random_forest.predict(scaled_data)
knn_predicted_price = knn.predict(scaled_data)
xgb_predicted_price = xgb.predict(scaled_data)

indr = float(input("Enter the Current 1$ Price in INDIAN Rupee:\n"))

lr_predicted_price *= indr
rfr_predicted_price *= indr
knn_predicted_price *= indr
xgb_predicted_price *= indr


print("Predicted Price from the Linear Regresion Model:",round(lr_predicted_price[0],2))
print("Predicted Price from the Randome Forest Regresson Model:",round(rfr_predicted_price[0],2))
print("Predicted Price from the KNN Model:",round(knn_predicted_price[0],2))
print("Predicted Price from the XG Boost Model:",round(xgb_predicted_price[0],2))


model = {
    "encoders": encoders,              
    "Standard_scaler": scl,            
    "Linear_Regression_model": lr,    
    "Random_Forest_regressor": random_forest,  
    "knn_model": knn,                
    "xgboost_model": xgb,              
    "columns": list(x.columns)         
    
}

with open("Models/laptop_price_prediction.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nPickle File Created Successfully!")