import streamlit as st
import pandas as pd
import pickle
import sklearn
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
import shap
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
    try:
        # Convertir columnas y eliminar irrelevantes
        df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'])


        # Transformar datos categóricos
        column_transformer = load('TAproject/column_transformer.joblib')
        data_transformed = column_transformer.transform(df)
        columns = column_transformer.get_feature_names_out()
        data_transformed_df = pd.DataFrame(data_transformed, columns=columns)

        columns_to_drop = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus',
                           'HomeOwnershipStatus', 'LoanPurpose', 'ApplicationDate',
                           'RiskScore','LoanApproved']
        data_cleaned = df.drop(columns=columns_to_drop)
        data_cleaned = pd.concat([data_cleaned, data_transformed_df], axis=1)
        data_cleaned = data_cleaned.apply(pd.to_numeric, errors='ignore')


        # Mostrar las primeras filas del dataframe final

        columns_to_drop2 = [col for col in data_cleaned.columns if col.startswith('remainder_')]
        data_cleaned = data_cleaned.drop(columns=columns_to_drop2)
        data_cleaned = data_cleaned.drop(columns=['MonthlyIncome'])
        columns_to_normalize = ['NetWorth',
                                'TotalDebtToIncomeRatio','AnnualIncome', 'LoanAmount'
                                , 'LoanDuration', 'MonthlyLoanPayment', 'Age', 'CreditScore',
                                'SavingsAccountBalance', 'CheckingAccountBalance', 'PaymentHistory'
                                ,'LengthOfCreditHistory','MonthlyDebtPayments']
        columns_to_normalize = [col for col in columns_to_normalize if col in data_cleaned.columns]

        scaler = load('TAproject/scaler.joblib')
        ##Aqui se cae
        data_cleaned[columns_to_normalize] = scaler.transform(data_cleaned[columns_to_normalize])






        return data_cleaned

    except Exception as e:
        st.error(f"Error durante el procesamiento de los datos: {e}")
        return None


# Función para análisis exploratorio de datos
def exploratory_data_analysis(data):
    st.write("## Análisis Exploratorio de Datos 📊")

    # Mostrar información general del conjunto de datos
    st.write("### Información General")
    st.write(f"**Número de filas:** {data.shape[0]}")
    st.write(f"**Número de columnas:** {data.shape[1]}")
    st.write("### Primeras filas del conjunto de datos:")
    st.dataframe(data.head())

    # Resumen estadístico
    st.write("### Resumen Estadístico")
    st.write(data.describe())

    # Filtrar columnas numéricas
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

    # Distribución de Variables
    st.write("### Distribución de Variables")
    selected_column = st.selectbox(
        "Selecciona una columna para analizar su distribución:",
        options=numeric_columns,
        key="distribution_column"
    )

    if selected_column:
        st.write(f"**Distribución de la columna:** {selected_column}")
        fig, ax = plt.subplots()
        data[selected_column].hist(bins=20, ax=ax, color="skyblue")
        ax.set_title(f"Distribución de {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

    # Comparación entre dos variables
    st.write("### Comparación entre dos variables")
    col1, col2 = st.columns(2)

    # Selección de las dos columnas para comparación
    with col1:
        column_x = st.selectbox(
            "Selecciona la columna para el eje X:",
            options=numeric_columns,
            key="x_column_compare"
        )
    with col2:
        column_y = st.selectbox(
            "Selecciona la columna para el eje Y:",
            options=numeric_columns,
            key="y_column_compare"
        )

    # Verificar que ambas columnas sean válidas
    if column_x and column_y:
        st.write(f"Comparando: **X = {column_x}**, **Y = {column_y}**")
        fig, ax = plt.subplots()
        ax.scatter(data[column_x], data[column_y], alpha=0.6, c="skyblue", edgecolor="k")
        ax.set_title(f"Comparación entre {column_x} y {column_y}")
        ax.set_xlabel(column_x)
        ax.set_ylabel(column_y)
        st.pyplot(fig)
# Streamlit Dashboard
def main():
    st.title("Dashboard de Riesgo Financiero para aprobación de prestamos bancario ")
    st.write("Este dashboard permite cargar datos de un cliente para la decision de brindarle un prestamo")

    # Paso 1: Cargar datos
    uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")
    data = load_data(uploaded_file)
    with open('TAproject/modelo_entrenado.pkl', 'rb') as file:
        modelo_cargado = pickle.load(file)
    if data is not None:
        # Seleccionar un registro específico
        exploratory_data_analysis(data)
        st.write("Selecciona el índice del registro a analizar:")
        registro_idx = st.number_input("Índice del registro", min_value=0, max_value=len(data) - 1, step=1, value=0)
        registro_especifico = data.iloc[[registro_idx]]  # Filtrar el registro seleccionado
        st.write("### Registro seleccionado:")
        st.dataframe(registro_especifico)

        if st.button("Pocesar  "):
            nueva_data_procesada = procesar_nueva_data(registro_especifico)
            prediccion = modelo_cargado.predict(nueva_data_procesada)
            probabilidad = modelo_cargado.predict_proba(nueva_data_procesada)

            # Mostrar los resultados
            st.write("### Resultado de la Predicción")
            st.write(f"Predicción: **{'Aprobado' if prediccion[0] == 1 else 'No Aprobado'}**")
            st.write(f"Probabilidad de Aprobación: **{probabilidad[0][1]:.2f}**")
            st.write(f"Probabilidad de No Aprobación: **{probabilidad[0][0]:.2f}**")
            # Paso 2: Visualizar importancia con SHAP
            st.write("### Explicación del modelo con SHAP")

            # Cargar datos y escalador necesarios para SHAP
            X_train_balanced = load('TAproject/X_train_balanced.joblib')  # Datos de entrenamiento balanceados usados
            explainer = shap.Explainer(modelo_cargado, X_train_balanced)

            # Seleccionar un registro específico del DataFrame procesado

            registro_especifico = nueva_data_procesada.iloc[[0]]

            # Calcular los valores SHAP
            shap_values = explainer(registro_especifico)

            # Mostrar gráfica de importancia con SHAP
            st.write(f"### Análisis SHAP para el registro {registro_idx}")
            st.write(f"Registro procesado:")
            st.dataframe(registro_especifico)

            fig, ax = plt.subplots()  # Crear una figura y un eje
            shap.waterfall_plot(shap_values[0, :, prediccion[0]])  # Dibujar el gráfico en el eje

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)  # Pasar la figura explícitamente


if __name__ == "__main__":
    main()