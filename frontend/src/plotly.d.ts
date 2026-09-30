declare module "plotly.js-dist-min" {
  export interface Figure {
    data: unknown[];
    layout: Record<string, unknown>;
  }
  const Plotly: {
    Plots: { resize(el: HTMLElement): Promise<void> };
    react(el: HTMLElement, figure: Figure): Promise<void>;
    purge(el: HTMLElement): void;
    toImage(
      el: HTMLElement,
      options: { format: "png"; width: number; height: number; scale: number },
    ): Promise<string>;
  };
  export default Plotly;
}
