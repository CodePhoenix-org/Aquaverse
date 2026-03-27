import { cn } from "../../lib/utils";
import OceanBackdrop from "./OceanBackdrop";

export default function PageShell({
  children,
  className,
  contentClassName,
  backdropVariant,
}) {
  return (
    <div className={cn("app-shell", className)}>
      <OceanBackdrop variant={backdropVariant} />
      <div className={cn("relative z-10 min-h-screen", contentClassName)}>
        {children}
      </div>
    </div>
  );
}
