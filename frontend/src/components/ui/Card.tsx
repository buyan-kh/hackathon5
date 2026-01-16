"use client";

import { forwardRef, HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const cardVariants = cva(
    "rounded-xl border bg-card text-foreground transition-all duration-200",
    {
        variants: {
            variant: {
                default: "border-border shadow-md hover:shadow-xl",
                elevated: "border-border shadow-lg hover:shadow-xl",
                featured: "border-2 border-transparent bg-gradient-to-br from-accent via-accent-secondary to-accent p-[2px]",
                flat: "border-border",
                ghost: "border-transparent bg-transparent",
            },
            hover: {
                true: "hover:bg-gradient-to-br hover:from-accent/[0.03] hover:to-transparent cursor-pointer",
                false: "",
            },
        },
        defaultVariants: {
            variant: "default",
            hover: false,
        },
    }
);

export interface CardProps
    extends HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof cardVariants> { }

const Card = forwardRef<HTMLDivElement, CardProps>(
    ({ className, variant, hover, children, ...props }, ref) => {
        if (variant === "featured") {
            return (
                <div className={cn(cardVariants({ variant }), className)} ref={ref} {...props}>
                    <div className="h-full w-full rounded-[calc(12px-2px)] bg-card p-6">
                        {children}
                    </div>
                </div>
            );
        }

        return (
            <div
                className={cn(cardVariants({ variant, hover }), "p-6", className)}
                ref={ref}
                {...props}
            >
                {children}
            </div>
        );
    }
);
Card.displayName = "Card";

const CardHeader = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
    ({ className, ...props }, ref) => (
        <div ref={ref} className={cn("flex flex-col space-y-1.5 pb-4", className)} {...props} />
    )
);
CardHeader.displayName = "CardHeader";

const CardTitle = forwardRef<HTMLHeadingElement, HTMLAttributes<HTMLHeadingElement>>(
    ({ className, ...props }, ref) => (
        <h3 ref={ref} className={cn("text-xl font-semibold tracking-tight", className)} {...props} />
    )
);
CardTitle.displayName = "CardTitle";

const CardDescription = forwardRef<HTMLParagraphElement, HTMLAttributes<HTMLParagraphElement>>(
    ({ className, ...props }, ref) => (
        <p ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />
    )
);
CardDescription.displayName = "CardDescription";

const CardContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
    ({ className, ...props }, ref) => (
        <div ref={ref} className={cn("", className)} {...props} />
    )
);
CardContent.displayName = "CardContent";

const CardFooter = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
    ({ className, ...props }, ref) => (
        <div ref={ref} className={cn("flex items-center pt-4", className)} {...props} />
    )
);
CardFooter.displayName = "CardFooter";

export { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter, cardVariants };
