import { cn } from "../../lib/utils";

const sizeMap = {
  sm: "h-10 w-10 rounded-xl",
  md: "h-12 w-12 rounded-2xl",
  lg: "h-16 w-16 rounded-3xl",
  xl: "h-24 w-24 rounded-[2rem]",
};

export default function AppLogo({
  size = "md",
  className,
  imageClassName,
  alt = "AquaVerse logo",
}) {
  return (
    <div
      className={cn(
        "overflow-hidden border border-white/[0.14] bg-gradient-to-br from-cyan-300/12 via-sky-300/10 to-blue-500/16 shadow-ocean",
        sizeMap[size] ?? sizeMap.md,
        className
      )}
    >
      <img
        src="/Aquaverse.jpeg"
        alt={alt}
        className={cn("h-full w-full object-cover", imageClassName)}
      />
    </div>
  );
}
