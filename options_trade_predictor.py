import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load your data
df = pd.read_excel('your_file.xlsx')

# Convert dates to datetime format and then to ordinal for numerical processing
df['Entry_Date'] = pd.to_datetime(df['Entry_Date']).map(dt.datetime.toordinal)
df['Exit_Date'] = pd.to_datetime(df['Exit_Date']).map(dt.datetime.toordinal)
df['Expiration_Date'] = pd.to_datetime(df['Expiration_Date']).map(dt.datetime.toordinal)

# Convert Ticker to numerical data using Label Encoding
df['Ticker'] = df['Ticker'].astype('category')
df['Ticker'] = df['Ticker'].cat.codes

# Separate features and target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid function for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=50, batch_size=10)
