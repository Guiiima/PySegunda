from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.datasets import load_iris

app = Flask(__name__)

# Carregando o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/treinar', methods=['POST'])
def treinar_testar():
    # Obtendo parâmetros do formulário
    tipo_classificador = request.form.get('tipo_classificador')
    param1 = request.form.get('param1')
    param2 = request.form.get('param2')

    # Verificando se o parâmetro obrigatório está presente
    if not param1 or not param1.strip():
        return render_template('parametros_faltando.html')

    try:
        # Convertendo parâmetros para tipos adequados
        param1 = float(param1)
        param2 = float(param2) if param2 else None
    except ValueError:
        return "Os parâmetros devem ser valores numéricos."

    classificador = None

    # Inicializando o classificador com base no tipo escolhido
    if tipo_classificador == 'KNN':
        classificador = KNeighborsClassifier(n_neighbors=int(param1))
    elif tipo_classificador == 'SVM':
        classificador = SVC(C=param1, kernel='linear')
    elif tipo_classificador == 'MLP':
        classificador = MLPClassifier(hidden_layer_sizes=(int(param1),), max_iter=int(param2))
    elif tipo_classificador == 'DT':
        classificador = DecisionTreeClassifier(max_depth=int(param1))
    else:
        return "Classificador não reconhecido."

    # Dividindo o conjunto de dados em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinando o classificador
    classificador.fit(X_treino, y_treino)

    # Realizando previsões no conjunto de teste
    y_predito = classificador.predict(X_teste)

    # Calculando métricas de desempenho
    acuracia = accuracy_score(y_teste, y_predito)
    precisao = precision_score(y_teste, y_predito, average='macro')
    recall = recall_score(y_teste, y_predito, average='macro')
    f1 = f1_score(y_teste, y_predito, average='macro')

    # Calculando a matriz de confusão
    cm = confusion_matrix(y_teste, y_predito)

    # Criando e personalizando o gráfico da matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Classes Reais')
    plt.xlabel('Classes Previstas')
    plt.title('Matriz de Confusão')
    plt.tight_layout()

    # Convertendo o gráfico em formato base64 para exibição na página web
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')

    # Exibindo os resultados na página web
    return render_template('result.html', accuracy=acuracia, precision=precisao, recall=recall, f1=f1, matriz_confusao=img_str)

if __name__ == '__main__':
    app.run(debug=True)
