import { cn } from "../../lib/utils";
import AppLogo from "./AppLogo";

export default function BrandMark({
  title = "AquaVerse",
  subtitle = "Ocean intelligence platform",
  compact = false,
  className,
}) {
  return (
    <div className={cn("flex items-center gap-3", className)}>
      <AppLogo size="md" />
      <div className="min-w-0">
        <p className="font-display text-lg font-bold tracking-[-0.03em] text-white">
          {title}
        </p>
        {!compact ? (
          <p className="truncate text-sm text-slate-300">{subtitle}</p>
        ) : null}
      </div>
    </div>
  );
}
