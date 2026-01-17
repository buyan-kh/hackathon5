"use client";

import { useEffect } from "react";

export function ForceLightMode() {
  useEffect(() => {
    // Remove any dark mode classes
    document.documentElement.classList.remove("dark");
    document.documentElement.removeAttribute("data-theme");
    
    // Force light mode
    document.documentElement.style.colorScheme = "light";
    document.body.style.backgroundColor = "#FAFAFA";
    document.body.style.color = "#0F172A";
    
    // Prevent dark mode detection
    const observer = new MutationObserver(() => {
      if (document.documentElement.classList.contains("dark")) {
        document.documentElement.classList.remove("dark");
      }
      if (document.documentElement.getAttribute("data-theme") === "dark") {
        document.documentElement.removeAttribute("data-theme");
      }
      document.documentElement.style.colorScheme = "light";
    });
    
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme"],
    });
    
    return () => observer.disconnect();
  }, []);
  
  return null;
}
