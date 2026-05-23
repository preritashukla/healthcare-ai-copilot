import * as React from "react"
import { cn } from "../../lib/utils"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost" | "critical"
  size?: "sm" | "md" | "lg"
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center font-medium rounded-md transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-pine focus-visible:ring-offset-2 active:scale-[0.98] active:translate-y-[1px] disabled:opacity-50 disabled:pointer-events-none disabled:active:translate-y-0",
          {
            // Primary
            "bg-pine text-white hover:bg-pine-light shadow-sm": variant === "primary",
            // Secondary
            "bg-slate-100 text-slate-800 hover:bg-slate-200": variant === "secondary",
            // Outline
            "border border-borderwhisper bg-transparent text-slate-700 hover:bg-slate-50": variant === "outline",
            // Ghost
            "bg-transparent text-slate-600 hover:bg-slate-50 hover:text-slate-900": variant === "ghost",
            // Critical Warning
            "bg-critical text-white hover:bg-red-800 shadow-sm": variant === "critical",
          },
          {
            "px-3 py-1.5 text-xs": size === "sm",
            "px-4 py-2 text-sm": size === "md",
            "px-5 py-2.5 text-base": size === "lg",
          },
          className
        )}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"
