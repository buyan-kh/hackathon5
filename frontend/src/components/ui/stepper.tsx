"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { IconCheck, IconLoader2, IconX } from "@tabler/icons-react";

interface StepperProps {
  steps: Array<{
    id: string;
    title: string;
    description?: string;
    status: "idle" | "running" | "completed" | "error";
    progress?: number;
  }>;
  orientation?: "horizontal" | "vertical";
  className?: string;
}

const Stepper = React.forwardRef<HTMLDivElement, StepperProps>(
  ({ steps, orientation = "vertical", className }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "flex",
          orientation === "horizontal" ? "flex-row" : "flex-col",
          className
        )}
      >
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={cn(
              "flex",
              orientation === "horizontal" ? "flex-col items-center" : "flex-row"
            )}
          >
            {/* Step Content */}
            <div className={cn("flex", orientation === "vertical" && "flex-1")}>
              {/* Indicator */}
              <div className="flex flex-col items-center">
                <div
                  className={cn(
                    "flex items-center justify-center rounded-full border-2 transition-all",
                    step.status === "completed"
                      ? "border-green-500 bg-green-500"
                      : step.status === "error"
                        ? "border-red-500 bg-red-500"
                        : step.status === "running"
                          ? "border-accent bg-accent/10"
                          : "border-muted-foreground/30 bg-background",
                    orientation === "horizontal" ? "h-10 w-10" : "h-8 w-8"
                  )}
                >
                  {step.status === "completed" ? (
                    <IconCheck size={16} className="text-white" />
                  ) : step.status === "error" ? (
                    <IconX size={16} className="text-white" />
                  ) : step.status === "running" ? (
                    <IconLoader2 size={16} className="text-accent animate-spin" />
                  ) : (
                    <span className="text-xs font-medium text-muted-foreground">
                      {index + 1}
                    </span>
                  )}
                </div>
                {/* Connector Line */}
                {index < steps.length - 1 && (
                  <div
                    className={cn(
                      orientation === "horizontal"
                        ? "w-full h-0.5 mt-2"
                        : "h-8 w-0.5 mt-2",
                      step.status === "completed"
                        ? "bg-green-500"
                        : "bg-muted-foreground/20"
                    )}
                  />
                )}
              </div>

              {/* Step Info */}
              <div className={cn("ml-4 pb-8", orientation === "horizontal" && "text-center ml-0 mt-2")}>
                <div className="flex items-center gap-2">
                  <h4
                    className={cn(
                      "text-sm font-medium",
                      step.status === "completed"
                        ? "text-green-600"
                        : step.status === "error"
                          ? "text-red-600"
                          : step.status === "running"
                            ? "text-accent"
                            : "text-muted-foreground"
                    )}
                  >
                    {step.title}
                  </h4>
                </div>
                {step.description && (
                  <p className="text-xs text-muted-foreground mt-1">{step.description}</p>
                )}
                {step.progress !== undefined && step.status === "running" && (
                  <div className="mt-2 w-full max-w-[200px] h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-accent to-accent-secondary transition-all duration-300"
                      style={{ width: `${step.progress}%` }}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }
);

Stepper.displayName = "Stepper";

export { Stepper };
