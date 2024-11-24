import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from joblib import load
# Función para cargar datos
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Vista previa de los datos cargados:")
        st.dataframe(data.head())
        return data
    else:
        st.warning("Por favor, sube un archivo CSV para continuar.")
        return None

# Preprocesamiento de datos
def preprocess_data(data, target_column):
    # Seleccionar las columnas más relevantes para evitar problemas de memoria
    selected_columns = [
        'Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration',
        'MonthlyDebtPayments', 'DebtToIncomeRatio', 'LoanPurpose', target_column
    ]
    data = data[selected_columns]

    # Convertir variables categóricas
    data = pd.get_dummies(data, columns=['LoanPurpose'], drop_first=True)

    # Separar características y etiqueta
    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y

# Función para balancear datos con SMOTE
def balance_data(X, y, random_state=42):
    if "sampling_strategy" not in st.session_state:
        st.session_state.sampling_strategy = 0.5  # Valor inicial por defecto

    sampling_strategy = st.slider(
        "Proporción de balanceo (sampling_strategy):",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.sampling_strategy,
        step=0.1,
        key="sampling_slider"
    )

    st.session_state.sampling_strategy = sampling_strategy

    st.write(f"Proporción de balanceo seleccionada: {sampling_strategy}")

    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    st.write("Distribución de clases después del balanceo:")
    st.write(pd.Series(y_balanced).value_counts())
    return X_balanced, y_balanced
# Función para entrenar el modelo
def train_model(X_train, y_train,modeloSeleccionado ,random_state=42):
    if modeloSeleccionado == "Random Forest":
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
    else:
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
    return model

# Función para evaluar el modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.subheader("Matriz de Confusión")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("Reporte de Clasificación")
    st.text(classification_report(y_test, y_pred))

    st.subheader("ROC-AUC Score")
    st.write(roc_auc_score(y_test, y_prob))


def procesar_nueva_data(df):
    # Eliminar columnas irrelevantes
    # Convertir la columna ApplicationDate a tipo datetime
    df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'])

    # Definir las columnas categóricas
    categorical_columns = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']

    # Crear el transformador de columnas
    column_transformer = load('column_transformer.joblib')

    # Aplicar la transformación
    data_transformed = column_transformer.transform(df)

    columns = column_transformer.get_feature_names_out()
    data_transformed_df = pd.DataFrame(data_transformed, columns=columns)

    # Eliminar las columnas originales del dataframe
    columns_to_drop = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose',
                       'ApplicationDate', 'RiskScore','LoanApproved']
    # data_cleaned = df.drop(columns=columns_to_drop
    data_cleaned = df.drop(columns=columns_to_drop)

    # Agregar las nuevas columnas transformadas al dataframe limpio
    data_cleaned = pd.concat([data_cleaned, data_transformed_df], axis=1)

    # Asegurarse de que todas las columnas sean numéricas
    data_cleaned = data_cleaned.apply(pd.to_numeric, errors='ignore')
    data_cleaned.head()
    # verficacion de la conversion de los datos
    numerical_columns = data_cleaned.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data_cleaned.select_dtypes(include=['object', 'category']).columns
    # Eliminar todas las columnas que comienzan con 'remainder_'
    columns_to_drop = [col for col in data_cleaned.columns if col.startswith('remainder_')]
    data_cleaned = data_cleaned.drop(columns=columns_to_drop)
    data_cleaned = data_cleaned.drop(columns=['MonthlyIncome'])
    # Mostrar las primeras filas del dataframe final
    # Con esto los valores
    columns_to_normalize = ['NetWorth', 'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'AnnualIncome', 'LoanAmount',
                            'LoanDuration', 'MonthlyLoanPayment', 'Age', 'CreditScore', 'SavingAccountBalance',
                            'CheckingAccountBalance', 'PaymentHistory', 'check_monthly_income', 'SavingsAccountBalance',
                            'LenghtOfCreditHistory', 'MonthlyDebtPayments']
    ## Asegurarse de que solo se normalicen las columnas que están presentes
    columns_to_normalize = [col for col in columns_to_normalize if col in data_cleaned.columns]

# Crear el escalador
    scaler = load('scaler.joblib')

# Normalizar las columnas seleccionadas
    data_cleaned[columns_to_normalize] = scaler.transform(data_cleaned[columns_to_normalize])

# Mostrar el dataframe normalizado


    # Crear DataFrame final

    return data_cleaned
# Streamlit Dashboard
def main():
    st.title("Dashboard de Riesgo Financiero para aprobación de ")
    st.write("Este dashboard permite cargar datos de un cliente para la decision de brindarle un prestamo")

    # Paso 1: Cargar datos
    uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")
    data = load_data(uploaded_file)
    with open('modelo_entrenado.pkl', 'rb') as file:
        modelo_cargado = pickle.load(file)
    if data is not None:

        if st.button("Pocesar  "):
            nueva_data_procesada = procesar_nueva_data(data)
            prediccion = modelo_cargado.predict(nueva_data_procesada)
            probabilidad = modelo_cargado.predict_proba(nueva_data_procesada)

            # Mostrar los resultados
            st.write("### Resultado de la Predicción")
            st.write(f"Predicción: **{'Aprobado' if prediccion[0] == 1 else 'No Aprobado'}**")
            st.write(f"Probabilidad de Aprobación: **{probabilidad[0][1]:.2f}**")
            st.write(f"Probabilidad de No Aprobación: **{probabilidad[0][0]:.2f}**")



if __name__ == "__main__":
    main()