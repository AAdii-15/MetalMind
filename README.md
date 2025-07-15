# 🧠 MetalMind – AI-Powered Price Forecasting for Indian Metals 🇮🇳

MetalMind is an AI-driven tool that predicts **metal prices (₹/kg)** for the next 10 days using:
- ✅ Real Indian market data
- 🌐 Latest business news sentiment
- 🔍 Custom logic (e.g., seasonal mining effects)

---

## 📌 Why MetalMind?

While interning at **Shri Ram Products** (a leading manufacturing firm), I realized small and medium manufacturers lack tools to make **informed buying decisions**.

To solve this, I created MetalMind – designed to **empower Indian manufacturers** with smarter forecasts.

---

## 🛠️ How It Works

1. **Input**: Choose a metal – copper, aluminium, steel, lead, zinc, or nickel
2. **Fetches**:
   - Live price data (e.g., from BankBazaar)
   - News articles from Reuters, Livemint, and Economic Times
3. **Analyzes**:
   - Sentiment using VADER (e.g., positive/negative news impact)
   - Rule-based logic (e.g., rain → mine shutdown → price rise)
4. **Predicts**:
   - Next 10 days of prices in ₹/kg using Random Forest model
5. **Visualizes**:
   - Forecast graph + date-wise prediction table

---

## 📊 Sample Output

Enter metal (copper, aluminium, steel, lead, zinc, nickel): aluminium
✅ Aluminium Current Price: ₹242.4 per kg (BankBazaar)

📅 10-Day Forecast (₹/kg):
2025-07-16 → ₹243.21
2025-07-17 → ₹244.32
...
2025-07-25 → ₹245.88

<img width="990" height="493" alt="Screenshot 2025-07-15 at 12 03 48 PM" src="https://github.com/user-attachments/assets/b5d8bdd3-5d1e-4d3d-aee9-ac838f023edf" />
## 🚀 Tech Stack

- Python (Pandas, Sklearn, Matplotlib)
- Web scraping (BeautifulSoup)
- News parsing (Feedparser)
- Sentiment analysis (VADER)
- Machine Learning: Random Forest Regressor

---

## 📁 Files

- `MetalMind.ipynb`: Main notebook
- `README.md`: Project description
- `requirements.txt` (optional): Dependencies

---

## 👨‍💻 Author

Adityaraj Dhuria  
Intern @ Shri Ram Products | BTech AI/ML | Aspiring Tech Entrepreneur

---

## 🧠 Future Improvements

- Add weather API to enhance seasonal logic
- Turn into full SaaS tool
- Mobile app for small industrialists

---

## 📬 Feedback

I'm open to collaborations or feedback – feel free to reach out!

