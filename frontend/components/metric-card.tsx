import { ReactNode } from "react";

type MetricCardProps = {
  label: string;
  value: ReactNode;
  accent?: "ember" | "moss" | "dusk";
  detail?: string;
  testId?: string;
};

const accentMap = {
  ember: "from-ember/15 to-ember/5 text-ember",
  moss: "from-moss/15 to-moss/5 text-moss",
  dusk: "from-dusk/15 to-dusk/5 text-dusk"
} as const;

export function MetricCard({
  label,
  value,
  accent = "dusk",
  detail,
  testId
}: MetricCardProps) {
  return (
    <div
      className={`min-w-0 rounded-3xl bg-gradient-to-br ${accentMap[accent]} p-[1px]`}
      data-testid={testId}
    >
      <div className="h-full min-w-0 rounded-[23px] bg-white/80 p-4">
        <p className="mono break-words text-xs uppercase tracking-[0.28em] text-ink/55">{label}</p>
        <div className="mt-3 min-w-0 break-words text-2xl font-semibold">{value}</div>
        {detail ? <p className="mt-2 break-words text-xs text-ink/50">{detail}</p> : null}
      </div>
    </div>
  );
}
