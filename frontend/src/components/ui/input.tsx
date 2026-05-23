import * as React from "react"
import { cn } from "../../lib/utils"

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  helperText?: string
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = "text", label, error, helperText, ...props }, ref) => {
    return (
      <div className="w-full flex flex-col gap-1.5">
        {label && (
          <label className="text-xs font-semibold text-slate-700 tracking-tight select-none">
            {label}
          </label>
        )}
        <input
          type={type}
          ref={ref}
          className={cn(
            "w-full px-3 py-2 text-sm bg-white border border-borderwhisper rounded-md transition-all duration-200 outline-none text-ink placeholder:text-slate-400 focus:border-pine focus:ring-1 focus:ring-pine disabled:bg-slate-50 disabled:text-slate-500 disabled:cursor-not-allowed",
            error && "border-critical focus:border-critical focus:ring-critical",
            className
          )}
          {...props}
        />
        {helperText && !error && (
          <p className="text-[11px] text-steel">{helperText}</p>
        )}
        {error && (
          <p className="text-[11px] text-critical font-medium">{error}</p>
        )}
      </div>
    )
  }
)
Input.displayName = "Input"
