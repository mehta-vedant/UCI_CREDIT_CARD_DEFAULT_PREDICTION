import Link from "next/link";

const stats = [
  { label: "ROC-AUC Score", value: "0.814" },
  { label: "Training Samples", value: "21,000" },
  { label: "Engineered Features", value: "17" },
  { label: "Models Compared", value: "6" },
];

const features = [
  { icon: "🔒", title: "Bank-Grade Accuracy", desc: "XGBoost trained on 30,000 real customer records with 81.4% ROC-AUC" },
  { icon: "⚡", title: "Instant Results", desc: "Get a risk assessment in under 2 seconds with probability scores" },
  { icon: "📊", title: "17 Smart Features", desc: "Engineered utilization ratios, payment behavior, and balance trends" },
  { icon: "🛡️", title: "Explainable Output", desc: "Clear risk levels with actionable recommendations, not just numbers" },
  { icon: "🧪", title: "6 Models Benchmarked", desc: "XGBoost, CatBoost, Random Forest, Gradient Boosting, AdaBoost, Logistic Regression" },
  { icon: "🎯", title: "Optimized Thresholds", desc: "F1-tuned classification threshold for balanced precision-recall" },
];

const steps = [
  { num: "01", title: "Enter Profile", desc: "Fill in credit, demographic, and payment history" },
  { num: "02", title: "AI Analysis", desc: "17 features engineered in real-time, fed to XGBoost" },
  { num: "03", title: "Get Risk Score", desc: "Probability gauge + risk level + what to do next" },
];

export default function LandingPage() {
  return (
    <main className="relative z-10 mx-auto max-w-6xl px-4 py-12">
      {/* Nav */}
      <nav className="mb-16 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-brand flex items-center justify-center text-white font-bold text-sm">CG</div>
          <span className="text-lg font-semibold">CreditGuard AI</span>
        </div>
        <Link
          href="/predict"
          className="rounded-full bg-brand px-5 py-2 text-sm font-medium text-white hover:bg-brand-light transition"
        >
          Get Your Score →
        </Link>
      </nav>

      {/* Hero */}
      <section className="mb-20 text-center">
        <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-brand">Predicting Default Before It Happens</p>
        <h1 className="mx-auto max-w-3xl text-4xl font-bold leading-tight md:text-6xl">
          <span className="gradient-text">Credit Default</span> Risk Prediction
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-base text-slate-400 md:text-lg">
          ML-powered classifier trained on 30,000 credit card records. Enter a customer profile and get an instant default-risk assessment with probability scores.
        </p>
        <div className="mt-8 flex justify-center gap-4">
          <Link href="/predict" className="rounded-full bg-brand px-8 py-3 text-sm font-semibold text-white hover:bg-brand-light transition shadow-lg shadow-brand-glow">
            Run Prediction
          </Link>
          <a href="#how-it-works" className="rounded-full border border-slate-600 px-8 py-3 text-sm font-semibold text-slate-300 hover:border-brand hover:text-brand transition">
            How It Works
          </a>
        </div>
      </section>

      {/* Stats strip */}
      <section className="mb-20 grid grid-cols-2 gap-4 md:grid-cols-4">
        {stats.map((s) => (
          <div key={s.label} className="rounded-2xl border border-dark-600 bg-dark-800 px-5 py-6 text-center">
            <p className="text-2xl font-bold text-brand">{s.value}</p>
            <p className="mt-1 text-xs text-slate-400">{s.label}</p>
          </div>
        ))}
      </section>

      {/* Features grid */}
      <section className="mb-20">
        <h2 className="mb-8 text-center text-2xl font-bold">Why CreditGuard AI</h2>
        <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <div key={f.title} className="rounded-2xl border border-dark-600 bg-dark-800 p-6 hover:border-brand/40 transition">
              <span className="text-2xl">{f.icon}</span>
              <h3 className="mt-3 font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="mb-20">
        <h2 className="mb-8 text-center text-2xl font-bold">How It Works</h2>
        <div className="grid gap-6 md:grid-cols-3">
          {steps.map((s) => (
            <div key={s.num} className="rounded-2xl border border-dark-600 bg-dark-800 p-8 text-center">
              <span className="inline-block rounded-full bg-brand/10 px-4 py-1 text-sm font-bold text-brand">{s.num}</span>
              <h3 className="mt-4 text-lg font-semibold">{s.title}</h3>
              <p className="mt-2 text-sm text-slate-400">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Model info */}
      <section className="mb-16 rounded-2xl border border-dark-600 bg-dark-800 p-8 text-center">
        <h2 className="text-xl font-bold">6 Models. One Winner.</h2>
        <p className="mx-auto mt-3 max-w-lg text-sm text-slate-400">
          We trained and tuned XGBoost, CatBoost, Random Forest, Gradient Boosting, AdaBoost, and Logistic Regression.
          XGBoost won with <span className="text-brand font-semibold">0.814 ROC-AUC</span> using GridSearchCV over 39 parameter combinations.
        </p>
        <Link href="/predict" className="mt-6 inline-block rounded-full bg-brand px-8 py-3 text-sm font-semibold text-white hover:bg-brand-light transition shadow-lg shadow-brand-glow">
          Try It Now →
        </Link>
      </section>

      {/* Footer */}
      <footer className="border-t border-dark-600 pt-6 text-center text-xs text-slate-500">
        Built with XGBoost + Next.js · Deployed on Vercel · Model trained on UCI Credit Card Default Dataset
      </footer>
    </main>
  );
}
