declare module "plotly.js-dist-min" {
  export interface Figure {
    data: unknown[];
    layout: Record<string, unknown>;
  }
  const Plotly: {
    react(el: HTMLElement, figure: Figure): Promise<void>;
    purge(el: HTMLElement): void;
  };
  export default Plotly;
}
