import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import yfinance as yf
from datetime import datetime
import pandas as pd


class PolynomialRegression():
    def __init__(self, x, y, degree=2, test_size=0.2):
        self.x = x
        self.y = y
        self.degree = degree
        self.test_size = test_size
        if len(x) != len(y):
            warnings.warn("x ve y eşit değil!")
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(x, y, test_size)

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
        print(f"R kare errorü = {r2}")
        return r2

    def mean_absolute_error(self):
        predictions = self.predict(self.x_test)
        mae = np.mean(np.abs(predictions - self.y_test))
        print(f"Ortalama Mutlak Hata (MAE): {mae}")
        return mae

    def mean_squared_error(self):
        predictions = self.predict(self.x_test)
        mse = np.mean((predictions - self.y_test) ** 2)
        print(f"Ortalama Kare Hata (MSE): {mse}")
        return mse

    def show_parameters(self):
        print(f"""Katsayılar: {self.theta_best.flatten()}""")

    def predict(self, x=None):
        if x is None:
            x = self.x
        x_poly = self.add_polynomial_features(x)
        prediction = x_poly.dot(self.theta_best)
        return prediction

    def plot(self, ticker_name, future_dates=None, future_predictions=None):
        plt.figure(figsize=(10, 6))
        plt.title(f"{ticker_name} Hisse Senedi Fiyat Tahmini")
        plt.xlabel("Tarih")
        plt.ylabel("Fiyat")

        x_dates = pd.to_datetime(data.index)
        y_prices = self.y.flatten()

        plt.scatter(x_dates, y_prices, color="r", label="Eğitim Verisi")
        plt.plot(x_dates, self.predict(), color="g", label="Tahmin")

        if future_dates is not None and future_predictions is not None:
            future_dates = pd.date_range(start=data.index[-1], periods=len(future_dates) + 1, closed='right')
            plt.plot(future_dates, future_predictions, color="m", linestyle="--", label="Gelecek Tahminleri")

        plt.legend()
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()


def fetch_data(ticker, start_date, end_date):
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(start=start_date, end=end_date)
    return data[['Close', 'Volume', 'Open', 'High', 'Low']]


def predict_future(regressor, days_into_future):
    last_date = regressor.x.max()
    future_dates = np.array(range(int(last_date) + 1, int(last_date) + 1 + days_into_future)).reshape(-1, 1)
    future_predictions = regressor.predict(future_dates)
    return future_dates, future_predictions


def get_valid_tickers():
    valid_tickers = ['ASELS.IS', 'AKBNK.IS', 'GARAN.IS', 'PETKM.IS', 'THYAO.IS', 'IZENR.IS', 'DOAS.IS','FROTO.IS',
                     'TTRAK.IS','EREGL.IS','SASA.IS']
    return valid_tickers


def get_user_choice(valid_tickers):
    print("Lütfen aşağıdaki hisselerden birini seçin:")
    for i, ticker in enumerate(valid_tickers, 1):
        print(f"{i}. {ticker}")
    choice = input("Seçiminiz: ")
    while not choice.isdigit() or int(choice) not in range(1, len(valid_tickers) + 1):
        print("Geçersiz giriş. Lütfen geçerli bir seçim yapın.")
        choice = input("Seçiminiz: ")
    return valid_tickers[int(choice) - 1]


def analyze_results(regressor):
    # R kare değerinin analizi
    r2 = regressor.r2_error()
    if r2 >= 0.8:
        r2_analysis = "yüksek"
    elif r2 >= 0.6:
        r2_analysis = "orta"
    elif r2 >= 0.4:
        r2_analysis = "orta düşük"
    else:
        r2_analysis = "düşük"

    # Ortalama Mutlak Hata ve Ortalama Kare Hata analizi
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

    # Hata histogramının analizi
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

    # Önerilerin oluşturulması
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
            f"Ortalama Kare Hata orta yüksek olduğu için modelin genelleştirme performansı düşük olabilir. "
            f"Regülerleştirme parametresini {regularization_decrease} azaltarak modelin performansını arttırabilirsiniz.")

    if mae_analysis == "yüksek":
        regularization_decrease = 0.5
        recommendations.append(
            f"Ortalama Mutlak Hata yüksek olduğu için modelin tahminleri gerçek değerlerden fazla sapabilir. "
            f"Daha fazla veri toplayarak veya regülerleştirme parametresini {regularization_decrease} "
            f"azaltarak modelin performansını arttırabilirsiniz.")
    elif mae_analysis == "orta yüksek":
        regularization_decrease = 0.3
        recommendations.append(
            f"Ortalama Mutlak Hata orta yüksek olduğu için modelin tahminleri gerçek değerlerden fazla sapabilir. "
            f"Regülerleştirme parametresini {regularization_decrease} azaltarak modelin performansını arttırabilirsiniz.")

    if error_analysis == "çok fazla negatif":
        degree_increase = 2
        recommendations.append(
            f"Hata histogramı, modelinizin tahminlerinin gerçek değerlerden {error_analysis} yönde sapma gösteriyor. "
            f"Bu durumu düzeltmek için polinom derecesini {degree_increase} arttırabilirsiniz.")
    elif error_analysis == "fazla negatif":
        degree_increase = 1
        recommendations.append(
            f"Hata histogramı, modelinizin tahminlerinin gerçek değerlerden {error_analysis} yönde sapma gösteriyor. "
            f"Bu durumu düzeltmek için polinom derecesini {degree_increase} arttırabilirsiniz.")
    elif error_analysis == "çok fazla pozitif":
        regularization_decrease = 0.3
        recommendations.append(
            f"Hata histogramı, modelinizin tahminlerinin gerçek değerlerden {error_analysis} yönde sapma gösteriyor. "
            f"Bu durumu düzeltmek için regülerleştirme parametresini {regularization_decrease} azaltabilirsiniz.")

    print("Analiz Sonuçları ve Öneriler:")
    print(f"- R kare değeri: {r2_analysis} ({r2})")
    print(f"- Ortalama Mutlak Hata (MAE): {mae_analysis} ({mae})")
    print(f"- Ortalama Kare Hata (MSE): {mse_analysis} ({mse})")
    print("- Öneriler:")
    for recommendation in recommendations:
        print(f"  - {recommendation}")


valid_tickers = get_valid_tickers()
chosen_ticker = get_user_choice(valid_tickers)
start_date = '2021-05-01'
end_date = datetime.now().strftime('%Y-%m-%d')
data = fetch_data(chosen_ticker, start_date, end_date)
dates = np.array(range(len(data))).reshape(-1, 1)
prices = data['Close'].values.reshape(-1, 1)
regressor = PolynomialRegression(dates, prices, degree=2)
regressor.normal_equation()
analyze_results(regressor)
future_dates, future_predictions = predict_future(regressor, 730)
regressor.plot(chosen_ticker, future_dates, future_predictions)
