import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import yfinance as yf
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PolynomialRegression:
    def __init__(self, x, y, degree=2, test_size=0.2):
        self.x = x
        self.y = y
        self.degree = degree
        self.test_size = test_size
        if len(x) != len(y):
            warnings.warn("x ve y eşit değil!")
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(x, y, test_size)
        self.theta_best = None

    def train_test_split(self, x, y, test_size):
        test_count = int(math.ceil(len(x) * test_size))
        data = np.c_[x, y]
        np.random.shuffle(data)
        x = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
        x_train = x[:-test_count]
        x_test = x[-test_count:]
        y_train = y[:-test_count]
        y_test = y[-test_count:]
        return x_train, x_test, y_train, y_test

    def add_polynomial_features(self, x):
        x_poly = np.ones((len(x), 1))
        for i in range(1, self.degree + 1):
            x_poly = np.c_[x_poly, x ** i]
        return x_poly

    def normal_equation(self, Lambda=0.03):
        x_poly = self.add_polynomial_features(self.x_train)
        reg_matrix = np.identity(len(x_poly[0]))
        reg_matrix[0][0] = 0
        self.theta_best = np.linalg.inv(x_poly.T.dot(x_poly) + reg_matrix * Lambda).dot(x_poly.T).dot(self.y_train)

    def r2_error(self):
        x_test_poly = self.add_polynomial_features(self.x_test)
        prediction = x_test_poly.dot(self.theta_best)
        RSS = np.sum((self.y_test - prediction) ** 2)
        TSS = np.sum((self.y_test - np.mean(self.y_test)) ** 2)
        r2 = 1 - (RSS / TSS)
        return r2

    def mean_absolute_error(self):
        predictions = self.predict(self.x_test)
        mae = np.mean(np.abs(predictions - self.y_test))
        return mae

    def mean_squared_error(self):
        predictions = self.predict(self.x_test)
        mse = np.mean((predictions - self.y_test) ** 2)
        return mse

    def show_parameters(self):
        print(f"Katsayılar: {self.theta_best.flatten()}")

    def predict(self, x=None):
        if self.theta_best is None:
            raise ValueError("Model henüz eğitilmedi. Lütfen önce normal_equation metodunu çağırın.")
        if x is None:
            x = self.x
        x_poly = self.add_polynomial_features(x)
        prediction = x_poly.dot(self.theta_best)
        return prediction

    def plot(self, ticker_name, data, future_dates=None, future_predictions=None):
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{ticker_name} Hisse Senedi Fiyat Tahmini", fontsize=16)
        ax.set_xlabel("Tarih", fontsize=14)
        ax.set_ylabel("Fiyat", fontsize=14)

        x_dates = pd.to_datetime(data.index)
        y_prices = self.y.flatten()

        ax.scatter(x_dates, y_prices, color="r", label="Eğitim Verisi", s=10)
        ax.plot(x_dates, self.predict(), color="g", label="Tahmin", linewidth=2)

        if future_dates is not None and future_predictions is not None:
            future_dates = pd.date_range(start=data.index[-1], periods=len(future_dates) + 1, closed='right')
            ax.plot(future_dates, future_predictions, color="m", linestyle="--", label="Gelecek Tahminleri", linewidth=2)

        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig

def fetch_data(ticker, start_date, end_date):
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(start=start_date, end=end_date)
    return data[['Close', 'Volume', 'Open', 'High', 'Low']]

def predict_future(regressor, days_into_future):
    last_date = regressor.x.max()
    future_dates = np.array(range(int(last_date) + 1, int(last_date) + 1 + days_into_future)).reshape(-1, 1)
    future_predictions = regressor.predict(future_dates)
    return future_dates, future_predictions

def analyze_results(regressor):
    r2 = regressor.r2_error()
    if (r2 >= 0.8):
        r2_analysis = "yüksek"
    elif (r2 >= 0.6):
        r2_analysis = "orta"
    elif (r2 >= 0.4):
        r2_analysis = "orta düşük"
    else:
        r2_analysis = "düşük"

    mae = regressor.mean_absolute_error()
    mse = regressor.mean_squared_error()

    if mae <= 5:
        mae_analysis = "düşük"
    elif mae <= 10:
        mae_analysis = "orta düşük"
    elif mae <= 15:
        mae_analysis = "orta yüksek"
    else:
        mae_analysis = "yüksek"

    if mse <= 50:
        mse_analysis = "düşük"
    elif mse <= 100:
        mse_analysis = "orta düşük"
    elif mse <= 150:
        mse_analysis = "orta yüksek"
    else:
        mse_analysis = "yüksek"

    errors = regressor.predict(regressor.x_test) - regressor.y_test
    error_mean = np.mean(errors)

    if error_mean < -5:
        error_analysis = "çok fazla negatif"
    elif error_mean < 0:
        error_analysis = "fazla negatif"
    elif error_mean == 0:
        error_analysis = "nötr"
    elif error_mean > 5:
        error_analysis = "çok fazla pozitif"
    else:
        error_analysis = "fazla pozitif"

    recommendations = []

    if r2_analysis == "düşük":
        degree_increase = 2
        recommendations.append(
            f"R kare değeri düşük olduğu için modelin daha karmaşık hale gelmesi gerekebilir. Polinom derecesini "
            f"{degree_increase} arttırarak modelin esnekliğini arttırabilirsiniz.")
    elif r2_analysis == "orta düşük":
        degree_increase = 1
        recommendations.append(
            f"R kare değeri orta düşük olduğu için modelin daha karmaşık hale gelmesi gerekebilir. Polinom derecesini "
            f"{degree_increase} arttırarak modelin esnekliğini arttırabilirsiniz.")
    elif r2_analysis == "orta":
        degree_increase = 1
        recommendations.append(
            f"R kare değeri orta düzeyde olduğu için modelin daha karmaşık hale gelmesi gerekebilir. Polinom derecesini"
            f"{degree_increase} arttırmak veya daha fazla veri toplamak modelin performansını arttırabilir.")
    elif r2_analysis == "yüksek":
        regularization_decrease = 0.3
        recommendations.append(
            f"R kare değeri yüksek olduğu için modelin regülerleştirme parametresini {regularization_decrease}"
            f" azaltarak modelin performansını arttırabilirsiniz.")

    if mse_analysis == "yüksek":
        regularization_decrease = 0.5
        recommendations.append(
            f"Ortalama Kare Hata yüksek olduğu için modelin genelleştirme performansı düşük olabilir."
            f" Daha fazla veri toplayarak veya regülerleştirme parametresini {regularization_decrease} "
            f"azaltarak modelin performansını arttırabilirsiniz.")
    elif mse_analysis == "orta yüksek":
        regularization_decrease = 0.3
        recommendations.append(
            f"Ortalama Kare Hata orta yüksek olduğu için modelin genelleştirme performansı arttırılabilir."
            f" Daha fazla veri toplayarak veya regülerleştirme parametresini {regularization_decrease} "
            f"azaltarak modelin performansını arttırabilirsiniz.")
    elif mse_analysis == "orta düşük":
        degree_increase = 1
        recommendations.append(
            f"Ortalama Kare Hata orta düşük olduğu için modelin polinom derecesini {degree_increase} "
            f"arttırarak performansını arttırabilirsiniz.")
    elif mse_analysis == "düşük":
        recommendations.append(
            f"Ortalama Kare Hata düşük olduğu için modelin performansı yeterli düzeyde gözükmektedir.")

    if mae_analysis == "yüksek":
        recommendations.append(
            f"Ortalama Mutlak Hata yüksek olduğu için modelin polinom derecesini arttırarak performansını "
            f"arttırabilirsiniz.")
    elif mae_analysis == "orta yüksek":
        degree_increase = 1
        recommendations.append(
            f"Ortalama Mutlak Hata orta yüksek olduğu için modelin polinom derecesini {degree_increase} "
            f"arttırarak performansını arttırabilirsiniz.")
    elif mae_analysis == "orta düşük":
        regularization_decrease = 0.1
        recommendations.append(
            f"Ortalama Mutlak Hata orta düşük olduğu için modelin regülerleştirme parametresini "
            f"{regularization_decrease} azaltarak performansını arttırabilirsiniz.")
    elif mae_analysis == "düşük":
        recommendations.append(
            f"Ortalama Mutlak Hata düşük olduğu için modelin performansı yeterli düzeyde gözükmektedir.")

    if error_analysis in ["çok fazla negatif", "çok fazla pozitif"]:
        recommendations.append(
            f"Modelin tahminlerinde sistematik bir hata var gibi gözüküyor. Veri setinizi kontrol edin ve "
            f"veri setindeki aykırı değerleri temizleyin. Ayrıca modelin polinom derecesini değiştirmeyi "
            f"de deneyebilirsiniz.")

    return {
        'r2': r2,
        'r2_analysis': r2_analysis,
        'mae': mae,
        'mae_analysis': mae_analysis,
        'mse': mse,
        'mse_analysis': mse_analysis,
        'error_mean': error_mean,
        'error_analysis': error_analysis,
        'recommendations': recommendations
    }

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hisse Senedi Tahmini")

        self.degree = 2
        self.Lambda = 0.03
        self.plot_window = None

        self.create_widgets()

    def create_widgets(self):
        # Başlık etiketi
        ttk.Label(self.root, text="Hisse Senedi Tahmini", font=("Helvetica", 16)).pack(pady=10)

        # Kullanıcıdan veri alma alanları
        self.ticker_entry = self.create_labeled_entry("Hisse Sembolü (örn: AAPL):")
        self.start_date_entry = self.create_labeled_entry("Başlangıç Tarihi (YYYY-MM-DD):")
        self.end_date_entry = self.create_labeled_entry("Bitiş Tarihi (YYYY-MM-DD):")
        self.days_into_future_entry = self.create_labeled_entry("Gelecekteki Gün Sayısı:")

        # Polinom derecesi ve Lambda giriş alanları
        self.degree_label = ttk.Label(self.root, text=f"Polinom Derecesi: {self.degree}")
        self.degree_label.pack(pady=5)
        self.degree_up_button = ttk.Button(self.root, text="Polinom Derecesini Arttır", command=self.increase_degree)
        self.degree_up_button.pack(pady=5)
        self.degree_down_button = ttk.Button(self.root, text="Polinom Derecesini Azalt", command=self.decrease_degree)
        self.degree_down_button.pack(pady=5)

        self.lambda_label = ttk.Label(self.root, text=f"Lambda: {self.Lambda}")
        self.lambda_label.pack(pady=5)
        self.lambda_up_button = ttk.Button(self.root, text="Lambda'yı Arttır", command=self.increase_lambda)
        self.lambda_up_button.pack(pady=5)
        self.lambda_down_button = ttk.Button(self.root, text="Lambda'yı Azalt", command=self.decrease_lambda)
        self.lambda_down_button.pack(pady=5)

        # Tahmin butonu
        self.predict_button = ttk.Button(self.root, text="Tahmin Yap", command=self.predict)
        self.predict_button.pack(pady=10)

    def create_labeled_entry(self, label_text):
        frame = ttk.Frame(self.root)
        frame.pack(pady=5)
        label = ttk.Label(frame, text=label_text)
        label.pack(side="left")
        entry = ttk.Entry(frame)
        entry.pack(side="left")
        return entry

    def increase_degree(self):
        self.degree += 1
        self.degree_label.config(text=f"Polinom Derecesi: {self.degree}")

    def decrease_degree(self):
        if (self.degree > 1):
            self.degree -= 1
            self.degree_label.config(text=f"Polinom Derecesi: {self.degree}")

    def increase_lambda(self):
        self.Lambda += 0.01
        self.lambda_label.config(text=f"Lambda: {self.Lambda}")

    def decrease_lambda(self):
        if (self.Lambda > 0.01):
            self.Lambda -= 0.01
            self.lambda_label.config(text=f"Lambda: {self.Lambda}")

    def predict(self):
        ticker = self.ticker_entry.get()
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()
        days_into_future = int(self.days_into_future_entry.get())

        try:
            self.data = fetch_data(ticker, start_date, end_date)
            y = self.data['Close'].values.reshape(-1, 1)
            x = np.array(range(1, len(y) + 1)).reshape(-1, 1)

            self.regressor = PolynomialRegression(x, y, degree=self.degree)
            self.regressor.normal_equation(self.Lambda)
            self.future_dates, self.future_predictions = predict_future(self.regressor, days_into_future)

            result = analyze_results(self.regressor)

            self.show_plot(ticker)
            self.show_analysis_results(result)

        except Exception as e:
            messagebox.showerror("Hata", f"Veri alınırken bir hata oluştu: {str(e)}")

    def show_plot(self, ticker_name):
        if self.plot_window is not None:
            self.plot_window.destroy()

        self.plot_window = tk.Toplevel(self.root)
        self.plot_window.title("Tahmin Grafiği")

        fig = self.regressor.plot(ticker_name, self.data, self.future_dates, self.future_predictions)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def show_analysis_results(self, result):
        result_text = (f"R^2 Değeri: {result['r2']} ({result['r2_analysis']})\n"
                       f"MAE: {result['mae']} ({result['mae_analysis']})\n"
                       f"MSE: {result['mse']} ({result['mse_analysis']})\n"
                       f"Hata Ortalaması: {result['error_mean']} ({result['error_analysis']})\n"
                       f"Tavsiyeler:\n")

        for recommendation in result['recommendations']:
            result_text += f"- {recommendation}\n"

        messagebox.showinfo("Analiz Sonuçları", result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()
