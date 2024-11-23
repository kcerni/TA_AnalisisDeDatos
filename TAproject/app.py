import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

# Streamlit Dashboard
def main():
    st.title("Dashboard de Entrenamiento de Modelos")
    st.write("Este dashboard permite cargar datos, balancearlos con SMOTE, entrenar un modelo de Random Forest y evaluar su desempeño.")

    # Paso 1: Cargar datos
    uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")
    data = load_data(uploaded_file)

    if data is not None:
        bins = st.slider("Número de bins en el histograma:", min_value=5, max_value=50, value=10, step=1)

        fig, ax = plt.subplots()
        data['CreditScore'].hist(bins=bins, ax=ax)
        ax.set_title("Histograma de Credit Score")
        ax.set_xlabel("Credit Score")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)
        # Paso 2: Seleccionar columna objetivo
        models = {
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=500)
        }
        target_column = st.selectbox("Selecciona la columna objetivo:", data.columns)
        modeloSeleccionado = st.selectbox("Selecciona el modelo:", models)
        x, y = preprocess_data(data, target_column)

        # Paso 3: Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


        # Paso 4: Balancear datos
        X_train_balanced, y_train_balanced = balance_data(X_train, y_train)
        if st.button("Pocesar y Entrenar Modelo "):
                st.write("Distribución de clases después de SMOTE:")
                st.write(pd.Series(y_train_balanced).value_counts())
                # Paso 5: Entrenar modelo
                model = train_model(X_train, y_train, modeloSeleccionado)
                st.success("Modelo entrenado con éxito.")

                # Paso 6: Evaluar modelo
                st.subheader("Evaluación del Modelo")
                evaluate_model(model, X_test, y_test)

                # Paso 7: Guardar modelo
        if st.button("Guardar Modelo"):
            joblib.dump(model, "rf_model.pkl")
            st.success("Modelo guardado como rf_model.pkl")

if __name__ == "__main__":
    main()