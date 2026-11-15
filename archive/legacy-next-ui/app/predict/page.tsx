"use client";
import { useState } from "react";
import Link from "next/link";

interface FormData {
  LIMIT_BAL: string; SEX: string; EDUCATION: string; MARRIAGE: string; AGE: string;
  PAY_0: string; PAY_2: string; PAY_3: string; PAY_4: string; PAY_5: string; PAY_6: string;
  BILL_AMT1: string; BILL_AMT2: string; BILL_AMT3: string; BILL_AMT4: string; BILL_AMT5: string; BILL_AMT6: string;
  PAY_AMT1: string; PAY_AMT2: string; PAY_AMT3: string; PAY_AMT4: string; PAY_AMT5: string; PAY_AMT6: string;
}

const defaultForm: FormData = {
  LIMIT_BAL: "20000", SEX: "1", EDUCATION: "1", MARRIAGE: "1", AGE: "30",
  PAY_0: "0", PAY_2: "0", PAY_3: "0", PAY_4: "0", PAY_5: "0", PAY_6: "0",
  BILL_AMT1: "10000", BILL_AMT2: "10000", BILL_AMT3: "10000", BILL_AMT4: "10000", BILL_AMT5: "10000", BILL_AMT6: "10000",
  PAY_AMT1: "1000", PAY_AMT2: "1000", PAY_AMT3: "1000", PAY_AMT4: "1000", PAY_AMT5: "1000", PAY_AMT6: "1000",
};

interface Result {
  prediction: string; probability: number; risk_level: string; risk_color: string;
}

const tips: Record<string, string[]> = {
  low: ["Keep paying on time", "Maintain credit utilization below 30%", "Avoid applying for multiple new accounts"],
  moderate: ["Focus on paying down existing balances", "Set up automatic payments to avoid missed dates", "Avoid new credit applications"],
  high: ["Prioritize reducing revolving balances immediately", "Contact your lender about hardship programs", "Avoid all new credit applications", "Consider balance transfer to lower interest"],
};

function RiskGauge({ color, prob }: { color: string; prob: number }) {
  const radius = 45;
  const circumference = Math.PI * radius;
  const offset = circumference - (prob / 100) * circumference;
  const strokeColor = color === "high" ? "#ef4444" : color === "moderate" ? "#f59e0b" : "#22c55e";

  return (
    <svg viewBox="0 0 100 60" className="mx-auto h-32 w-52">
      <path d="M 5 55 A 45 45 0 0 1 95 55" fill="none" stroke="#1e293b" strokeWidth="10" strokeLinecap="round" />
      <path
        d="M 5 55 A 45 45 0 0 1 95 55" fill="none" stroke={strokeColor} strokeWidth="10" strokeLinecap="round"
        strokeDasharray={circumference} strokeDashoffset={offset}
        style={{ animation: "gauge-fill 1s ease forwards", filter: `drop-shadow(0 0 6px ${strokeColor})` }}
      />
      <text x="50" y="48" textAnchor="middle" fill={strokeColor} fontSize="16" fontWeight="bold">
        {prob}%
      </text>
      <text x="50" y="57" textAnchor="middle" fill="#94a3b8" fontSize="5">
        default probability
      </text>
    </svg>
  );
}

function Tooltip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <span className="relative ml-1 inline-block">
      <button
        type="button"
        className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-slate-600 text-[10px] text-slate-400 hover:border-brand hover:text-brand transition"
        onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}
        onClick={() => setShow(!show)}
      >?</button>
      {show && (
        <span className="absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-dark-600 bg-dark-800 px-3 py-2 text-xs text-slate-300 shadow-xl">
          {text}
        </span>
      )}
    </span>
  );
}

const fieldLabels: Record<keyof FormData, [string, string]> = {
  LIMIT_BAL: ["Credit Limit", "Total approved credit in NT dollars"],
  SEX: ["Sex", "1 = Male, 2 = Female"],
  EDUCATION: ["Education", "1 = Grad, 2 = Uni, 3 = HS, 4 = Other"],
  MARRIAGE: ["Marital Status", "1 = Married, 2 = Single, 3 = Other"],
  AGE: ["Age", "Years"],
  PAY_0: ["Repayment Status (Sep)", "-1=On time, 1=1mo late, 2=2mo late..."],
  PAY_2: ["Repayment Status (Aug)", "Same scale as above"],
  PAY_3: ["Repayment Status (Jul)", "Same scale as above"],
  PAY_4: ["Repayment Status (Jun)", "Same scale as above"],
  PAY_5: ["Repayment Status (May)", "Same scale as above"],
  PAY_6: ["Repayment Status (Apr)", "Same scale as above"],
  BILL_AMT1: ["Bill Amount (Sep)", "NT dollars"],
  BILL_AMT2: ["Bill Amount (Aug)", "NT dollars"],
  BILL_AMT3: ["Bill Amount (Jul)", "NT dollars"],
  BILL_AMT4: ["Bill Amount (Jun)", "NT dollars"],
  BILL_AMT5: ["Bill Amount (May)", "NT dollars"],
  BILL_AMT6: ["Bill Amount (Apr)", "NT dollars"],
  PAY_AMT1: ["Payment Amount (Sep)", "NT dollars"],
  PAY_AMT2: ["Payment Amount (Aug)", "NT dollars"],
  PAY_AMT3: ["Payment Amount (Jul)", "NT dollars"],
  PAY_AMT4: ["Payment Amount (Jun)", "NT dollars"],
  PAY_AMT5: ["Payment Amount (May)", "NT dollars"],
  PAY_AMT6: ["Payment Amount (Apr)", "NT dollars"],
};

export default function PredictPage() {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<FormData>(defaultForm);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = (k: keyof FormData, v: string) => setForm((prev) => ({ ...prev, [k]: v }));

  const stepFields: (keyof FormData)[][] = [
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"],
    ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
     "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
  ];
  const stepTitles = ["Profile & Credit Limit", "Payment History", "Bill & Payment Amounts"];

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      setResult(await res.json());
    } catch (e: any) {
      setError(e.message || "Failed to get prediction");
    } finally {
      setLoading(false);
    }
  };

  if (result) {
    return (
      <main className="relative z-10 mx-auto max-w-2xl px-4 py-12">
        <Link href="/" className="mb-8 inline-flex items-center gap-2 text-sm text-slate-400 hover:text-brand transition">
          ← Back to Home
        </Link>
        <h1 className="mb-6 text-2xl font-bold">Prediction Result</h1>

        <div className="rounded-2xl border border-dark-600 bg-dark-800 p-8 text-center animate-slide-up">
          <RiskGauge color={result.risk_color} prob={result.probability} />
          <p className="mt-4 text-xl font-bold" style={{ color: result.risk_color === "high" ? "#ef4444" : result.risk_color === "moderate" ? "#f59e0b" : "#22c55e" }}>
            {result.risk_level}
          </p>
          <p className="mt-1 text-sm text-slate-400">{result.prediction}</p>
        </div>

        {/* Input summary */}
        <div className="mt-6 rounded-2xl border border-dark-600 bg-dark-800 p-6">
          <h3 className="mb-3 text-sm font-semibold text-slate-300">Input Summary</h3>
          <div className="grid grid-cols-2 gap-3 text-xs">
            {(["LIMIT_BAL", "AGE", "EDUCATION", "SEX", "MARRIAGE"] as (keyof FormData)[]).map((k) => (
              <div key={k} className="flex justify-between rounded-lg bg-dark-700 px-3 py-2">
                <span className="text-slate-400">{fieldLabels[k][0]}</span>
                <span className="font-medium">{form[k]}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Recommendations */}
        <div className="mt-6 rounded-2xl border border-dark-600 bg-dark-800 p-6">
          <h3 className="mb-3 text-sm font-semibold text-slate-300">Recommendations</h3>
          <ul className="space-y-2">
            {tips[result.risk_color]?.map((t, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-400">
                <span className="mt-0.5 text-brand">›</span>{t}
              </li>
            ))}
          </ul>
        </div>

        <button onClick={() => setResult(null)} className="mt-6 w-full rounded-full bg-brand py-3 text-sm font-semibold text-white hover:bg-brand-light transition">
          Run Another Prediction
        </button>
      </main>
    );
  }

  return (
    <main className="relative z-10 mx-auto max-w-2xl px-4 py-12">
      <Link href="/" className="mb-8 inline-flex items-center gap-2 text-sm text-slate-400 hover:text-brand transition">
        ← Back to Home
      </Link>

      {/* Progress */}
      <div className="mb-8">
        <div className="flex items-center gap-3 text-xs text-slate-500">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex items-center gap-2">
              <div className={`flex h-7 w-7 items-center justify-center rounded-full border text-xs font-bold transition
                ${step === i ? "border-brand bg-brand text-white" : step > i ? "border-brand bg-brand/20 text-brand" : "border-dark-600 bg-dark-700 text-slate-500"}`}>
                {i + 1}
              </div>
              <span className={step === i ? "text-white" : ""}>{stepTitles[i]}</span>
              {i < 2 && <div className="mx-2 h-px w-8 bg-dark-600" />}
            </div>
          ))}
        </div>
      </div>

      <h1 className="mb-2 text-2xl font-bold">New Prediction</h1>
      <p className="mb-6 text-sm text-slate-400">Step {step + 1} of 3 — {stepTitles[step]}</p>

      {/* Fields */}
      <div className="space-y-4">
        {stepFields[step].map((k) => {
          const [label, tooltip] = fieldLabels[k];
          const isSelect = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"].includes(k);
          return (
            <div key={k} className="animate-fade-in">
              <label className="mb-1 block text-xs font-medium text-slate-300">
                {label} <Tooltip text={tooltip} />
              </label>
              {isSelect ? (
                <select
                  value={form[k]} onChange={(e) => set(k, e.target.value)}
                  className="w-full rounded-xl border border-dark-600 bg-dark-700 px-4 py-3 text-sm text-white focus:border-brand focus:outline-none transition"
                >
                  {k.startsWith("PAY_") && (
                    <>
                      <option value="-2">No consumption</option>
                      <option value="-1">Paid in full</option>
                      <option value="0">Revolving credit</option>
                      <option value="1">1 month late</option>
                      <option value="2">2 months late</option>
                      <option value="3">3 months late</option>
                      <option value="4">4+ months late</option>
                    </>
                  )}
                  {k === "SEX" && <><option value="1">Male</option><option value="2">Female</option></>}
                  {k === "EDUCATION" && (
                    <><option value="1">Graduate School</option><option value="2">University</option><option value="3">High School</option><option value="4">Other</option></>
                  )}
                  {k === "MARRIAGE" && <><option value="1">Married</option><option value="2">Single</option><option value="3">Other</option></>}
                </select>
              ) : (
                <input
                  type="number" value={form[k]} onChange={(e) => set(k, e.target.value)}
                  className="w-full rounded-xl border border-dark-600 bg-dark-700 px-4 py-3 text-sm text-white focus:border-brand focus:outline-none transition"
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Navigation */}
      <div className="mt-8 flex items-center justify-between">
        <button
          onClick={() => setStep((s) => s - 1)} disabled={step === 0}
          className="rounded-full border border-dark-600 px-6 py-3 text-sm text-slate-400 hover:border-brand hover:text-brand transition disabled:opacity-30"
        >
          Back
        </button>
        {step < 2 ? (
          <button
            onClick={() => setStep((s) => s + 1)}
            className="rounded-full bg-brand px-8 py-3 text-sm font-semibold text-white hover:bg-brand-light transition"
          >
            Next →
          </button>
        ) : (
          <button
            onClick={handleSubmit} disabled={loading}
            className="rounded-full bg-brand px-8 py-3 text-sm font-semibold text-white hover:bg-brand-light transition shadow-lg shadow-brand-glow disabled:opacity-50"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="32" strokeLinecap="round" /></svg>
                Analyzing...
              </span>
            ) : "Get Prediction"}
          </button>
        )}
      </div>

      {error && <p className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-center text-sm text-red-400">{error}</p>}
    </main>
  );
}
