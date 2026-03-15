type FunctionCardProps = {
  title: string;
  mode: "Live" | "Static";
  description: string;
};

export function FunctionCard({ title, mode, description }: FunctionCardProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">{title}</h3>
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-medium ${
            mode === "Live"
              ? "bg-sky-100 text-sky-700"
              : "bg-orange-100 text-orange-700"
          }`}
        >
          {mode}
        </span>
      </div>
      <p className="text-xs text-slate-600">{description}</p>
    </div>
  );
}
