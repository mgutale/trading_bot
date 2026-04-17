"""
Dashboard Theme

Single source of truth for all dashboard colors and styling.
Used by both Python components and CSS.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DashboardTheme:
    """
    Centralized theme configuration for the trading bot dashboard.

    All colors follow a semantic naming convention:
    - success: Positive outcomes (profits, gains)
    - danger: Negative outcomes (losses, drawdowns)
    - warning: Caution states (weak regimes, volatility)
    - primary: Main accent color
    - info: Secondary information

    Usage:
        theme = DashboardTheme()
        color = theme.success  # '#3fb950'
    """

    # Backgrounds
    bg_primary: str = "#0a0a0a"       # Main background
    bg_secondary: str = "#161b22"     # Secondary background
    bg_card: str = "#21262d"          # Card backgrounds
    bg_hover: str = "#30363d"         # Hover states

    # Text
    text_primary: str = "#f0f6fc"     # Primary text
    text_secondary: str = "#8b949e"   # Secondary text
    text_muted: str = "#6e7681"       # Muted/disabled text

    # Borders
    border_default: str = "#30363d"   # Default borders
    border_subtle: str = "#21262d"    # Subtle borders
    border_focus: str = "#58a6ff"     # Focus states

    # Semantic colors
    success: str = "#3fb950"          # Positive P&L, strong bull
    danger: str = "#f85149"           # Negative P&L, strong bear
    warning: str = "#d29922"          # Weak bear, caution
    primary: str = "#58a6ff"          # Primary accent
    info: str = "#79c0ff"             # Secondary info
    purple: str = "#bf5af2"           # Additional accent

    # Regime-specific colors
    regime_colors: Dict[str, str] = field(default_factory=lambda: {
        'strong_bull': '#006400',     # Dark green
        'weak_bull': '#2ca02c',       # Green
        'weak_bear': '#ff7f0e',       # Orange
        'strong_bear': '#8b0000',     # Dark red
        'unknown': '#808080',         # Gray
    })

    # Chart color palette (ordered for multi-series)
    chart_palette: List[str] = field(default_factory=lambda: [
        '#58a6ff',  # Blue - primary series
        '#3fb950',  # Green - secondary series
        '#ff7f0e',  # Orange - tertiary
        '#f85149',  # Red - quaternary
        '#bf5af2',  # Purple - fifth
        '#79c0ff',  # Light blue
        '#2ca02c',  # Dark green
        '#d29922',  # Dark orange
        '#a371f7',  # Light purple
        '#ff9492',  # Light red
    ])

    # Heatmap colors (diverging: negative -> neutral -> positive)
    heatmap_colors: List[List[float]] = field(default_factory=lambda: [
        [0.0, '#d62728'],    # Red for losses
        [0.5, '#ffdd00'],    # Yellow for neutral
        [1.0, '#2ca02c'],    # Green for gains
    ])

    # Typography
    font_family: str = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    font_size_base: str = "14px"
    font_size_small: str = "12px"
    font_size_large: str = "18px"
    font_size_xlarge: str = "24px"

    # Spacing scale (4px base unit)
    spacing_xs: str = "4px"
    spacing_sm: str = "8px"
    spacing_md: str = "16px"
    spacing_lg: str = "24px"
    spacing_xl: str = "32px"

    # Border radius
    radius_sm: str = "4px"
    radius_md: str = "8px"
    radius_lg: str = "12px"
    radius_xl: str = "16px"
    radius_full: str = "9999px"

    # Chart defaults
    chart_height_sm: int = 250
    chart_height_md: int = 350
    chart_height_lg: int = 450
    chart_line_width: int = 2

    def get_regime_color(self, regime: str) -> str:
        """Get color for a market regime."""
        return self.regime_colors.get(regime, self.regime_colors['unknown'])

    def get_chart_color(self, index: int) -> str:
        """Get color from chart palette by index (cycles)."""
        return self.chart_palette[index % len(self.chart_palette)]

    def color_for_return(self, value: float) -> str:
        """Get color based on return value (positive=success, negative=danger)."""
        if value > 0:
            return self.success
        elif value < 0:
            return self.danger
        return self.text_secondary

    def to_css_variables(self) -> str:
        """
        Generate CSS custom properties from theme.

        Usage in CSS:
            :root {
                {theme.to_css_variables()}
            }
        """
        return f"""--bg-primary: {self.bg_primary};
    --bg-secondary: {self.bg_secondary};
    --bg-card: {self.bg_card};
    --bg-hover: {self.bg_hover};
    --text-primary: {self.text_primary};
    --text-secondary: {self.text_secondary};
    --text-muted: {self.text_muted};
    --border-default: {self.border_default};
    --border-subtle: {self.border_subtle};
    --border-focus: {self.border_focus};
    --success: {self.success};
    --danger: {self.danger};
    --warning: {self.warning};
    --primary: {self.primary};
    --info: {self.info};
    --purple: {self.purple};
    --font-family: {self.font_family};
    --font-size-base: {self.font_size_base};
    --font-size-small: {self.font_size_small};
    --font-size-large: {self.font_size_large};
    --spacing-xs: {self.spacing_xs};
    --spacing-sm: {self.spacing_sm};
    --spacing-md: {self.spacing_md};
    --spacing-lg: {self.spacing_lg};
    --spacing-xl: {self.spacing_xl};
    --radius-sm: {self.radius_sm};
    --radius-md: {self.radius_md};
    --radius-lg: {self.radius_lg};
    --radius-xl: {self.radius_xl};"""


# Global theme instance for import convenience
DEFAULT_THEME = DashboardTheme()
