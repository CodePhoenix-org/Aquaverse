import { cn } from "../../lib/utils";

const variantMap = {
  default: {
    glow: "bg-cyan-400/12",
    secondary: "bg-sky-500/10",
  },
  dense: {
    glow: "bg-cyan-300/16",
    secondary: "bg-blue-500/14",
  },
};

export default function OceanBackdrop({
  className,
  variant = "default",
  withGrid = true,
}) {
  const theme = variantMap[variant] ?? variantMap.default;

  return (
    <div
      aria-hidden="true"
      className={cn("pointer-events-none absolute inset-0 overflow-hidden", className)}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(56,189,248,0.08),transparent_36%),linear-gradient(180deg,rgba(2,10,23,0),rgba(2,10,23,0.4))]" />
      {withGrid ? <div className="premium-grid absolute inset-0 opacity-70" /> : null}
      <div
        className={cn(
          "absolute -top-40 left-1/2 h-[34rem] w-[34rem] -translate-x-1/2 rounded-full blur-[140px]",
          theme.glow
        )}
      />
      <div
        className={cn(
          "absolute -right-16 top-24 h-[24rem] w-[24rem] rounded-full blur-[120px]",
          theme.secondary
        )}
      />
      <div className="absolute -left-24 bottom-0 h-[22rem] w-[22rem] rounded-full bg-teal-400/10 blur-[120px]" />
      <div className="absolute inset-x-0 bottom-0 h-[38vh] bg-[radial-gradient(ellipse_at_bottom,rgba(14,165,233,0.16),transparent_68%)]" />
    </div>
  );
}
