import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Veri üretimi
np.random.seed(42)
n = 200
tecrube = np.random.uniform(0, 10, n)
teknik_puan = np.random.uniform(0, 100, n)
etiket = [1 if t < 2 and p < 60 else 0 for t, p in zip(tecrube, teknik_puan)]

df = pd.DataFrame({
    'tecrube_yili': tecrube,
    'teknik_puan': teknik_puan,
    'etiket': etiket
})

print("Etiket dağilimi:\n", df['etiket'].value_counts())

# 2. Eğitim/test ayırma
X = df[['tecrube_yili', 'teknik_puan']]
y = df['etiket']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. SVM modeli (linear)
model_linear = SVC(kernel='linear')
model_linear.fit(X_train_scaled, y_train)

# 5. Karar sınırı görselleştirme fonksiyonu
def plot_decision_boundary(model, X, y, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=60, alpha=0.7)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, linewidth=1.5, facecolors='none', edgecolors='k')
    plt.title(title)
    plt.xlabel("Tecrübe (scaled)")
    plt.ylabel("Teknik Puan (scaled)")
    plt.grid(True)
    plt.show()

plot_decision_boundary(model_linear, X_train_scaled, y_train.to_numpy(), "Linear SVM - Karar Siniri")

# 6. Tahmin (sabit değerle)
tecrube_input = 1.5
puan_input = 55
input_scaled = scaler.transform([[tecrube_input, puan_input]])
tahmin = model_linear.predict(input_scaled)
print(f"\nTahmin için giriş: {tecrube_input} yil tecrübe, {puan_input} puan")
print("Linear SVM tahmin sonucu:", "İşe alinmadi " if tahmin[0] == 1 else "İşe alindi ")

# 7. Model değerlendirmesi
y_pred = model_linear.predict(X_test_scaled)
print("\n--- Linear SVM Değerlendirme ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#  8. Kernel değişimi - RBF SVM
model_rbf = SVC(kernel='rbf')
model_rbf.fit(X_train_scaled, y_train)
plot_decision_boundary(model_rbf, X_train_scaled, y_train.to_numpy(), "RBF (Doğrusal Olmayan) SVM - Karar Siniri")

# 9. Parametre tuning (Grid Search)
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 0.01],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("\n--- En iyi parametreler (GridSearchCV) ---")
print(grid.best_params_)
best_model = grid.best_estimator_
best_pred = best_model.predict(X_test_scaled)
print("\n--- GridSearchCV Sonuçlari ---")
print("Accuracy:", accuracy_score(y_test, best_pred))
print("Classification Report:\n", classification_report(y_test, best_pred))

import joblib

joblib.dump(model_linear, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
