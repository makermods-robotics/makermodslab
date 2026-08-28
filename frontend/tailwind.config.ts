import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1440px'
			}
		},
		extend: {
			fontFamily: {
				display: ['"Inter"', 'system-ui', 'sans-serif'],
				body: ['"Inter"', 'system-ui', 'sans-serif'],
				mono: ['"JetBrains Mono"', 'ui-monospace', 'Menlo', 'monospace'],
				orbitron: ['"Orbitron"', 'system-ui', 'sans-serif'],
			},
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				brand: {
					DEFAULT: 'hsl(var(--brand))',
					foreground: 'hsl(var(--brand-foreground))',
					hover: 'hsl(var(--brand-hover))',
					tint: 'hsl(var(--brand-tint))'
				},
				ok: 'hsl(var(--ok))',
				warn: 'hsl(var(--warn))',
				info: 'hsl(var(--info))'
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			boxShadow: {
				'1': 'var(--shadow-1)',
				'2': 'var(--shadow-2)'
			},
			transitionTimingFunction: {
				std: 'cubic-bezier(0.2, 0, 0, 1)'
			},
			keyframes: {
				'accordion-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-accordion-content-height)'
					}
				},
				'accordion-up': {
					from: {
						height: 'var(--radix-accordion-content-height)'
					},
					to: {
						height: '0'
					}
				},
				'collapsible-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-collapsible-content-height)'
					}
				},
				'collapsible-up': {
					from: {
						height: 'var(--radix-collapsible-content-height)'
					},
					to: {
						height: '0'
					}
				},
				'spotlight-pulse': {
					'0%, 100%': {
						boxShadow: '0 0 0 0 hsl(var(--ring) / 0.55)',
						transform: 'scale(1)'
					},
					'50%': {
						boxShadow: '0 0 0 8px hsl(var(--ring) / 0)',
						transform: 'scale(1.04)'
					}
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'collapsible-down': 'collapsible-down 0.25s cubic-bezier(0.2, 0, 0, 1)',
				'collapsible-up': 'collapsible-up 0.2s cubic-bezier(0.2, 0, 0, 1)',
				'spotlight-pulse': 'spotlight-pulse 1.8s cubic-bezier(0.2, 0, 0, 1) infinite'
			}
		}
	},
	plugins: [
		require("tailwindcss-animate"),
		require("@tailwindcss/container-queries"),
	],
} satisfies Config;
