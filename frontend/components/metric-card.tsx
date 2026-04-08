type MetricCardProps = {
  label: string;
  value: string;
  accent?: "ember" | "moss" | "dusk";
};

const accentMap = {
  ember: "from-ember/15 to-ember/5 text-ember",
  moss: "from-moss/15 to-moss/5 text-moss",
  dusk: "from-dusk/15 to-dusk/5 text-dusk"
} as const;

export function MetricCard({
  label,
  value,
  accent = "dusk"
}: MetricCardProps) {
  return (
    <div className={`rounded-3xl bg-gradient-to-br ${accentMap[accent]} p-[1px]`}>
      <div className="h-full rounded-[23px] bg-white/80 p-4">
        <p className="mono text-xs uppercase tracking-[0.28em] text-ink/55">{label}</p>
        <p className="mt-3 text-2xl font-semibold">{value}</p>
      </div>
    </div>
  );
}
