from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import numpy as np

def preprocess_data(data, target_column, save_path, output_path):
    # Drop duplikat sebelum preprocessing
    print("Jumlah duplikasi sebelum drop:", data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    print("Jumlah duplikasi setelah drop:", data.duplicated().sum())

    # Menentukan fitur numerik dan kategoris
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    
    # Pastikan target_column tidak masuk ke fitur
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)
    
    # Pipeline untuk fitur numerik (imputer + scaler)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk fitur kategorikal (imputer + onehotencoder)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Gabungkan pipeline dengan ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Pisahkan fitur dan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Label encoding untuk target jika target berupa kategori/string
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Target '{target_column}' sudah di-label encode.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    # Fitting dan transformasi pada train
    X_train_transformed = preprocessor.fit_transform(X_train)
    # Transformasi pada test
    X_test_transformed = preprocessor.transform(X_test)

    # Gabungkan X_train dan y_train
    train_data = np.hstack((X_train_transformed, y_train.reshape(-1, 1)))

    # Ambil nama-nama kolom hasil transformasi
    feature_names = preprocessor.get_feature_names_out()
    all_columns = list(feature_names) + [target_column]

    # Simpan ke file CSV
    pd.DataFrame(train_data, columns=all_columns).to_csv(output_path, index=False)
    print(f"Dataset hasil preprocessing disimpan ke: {output_path}")

    # Simpan pipeline preprocessing ke file
    dump(preprocessor, save_path)
    print(f"Preprocessing pipeline disimpan ke: {save_path}")

    return X_train_transformed, X_test_transformed, y_train, y_test

if __name__ == "__main__":
    target_column = "Personality"
    save_path = "preprocessing/preprocessor_pipeline.joblib"
    file_path = "personality_dataset_raw.csv"
    output_path = "preprocessing/personality_dataset_preprocessing.csv"

    data = pd.read_csv(file_path)
    preprocess_data(data, target_column, save_path, output_path)
