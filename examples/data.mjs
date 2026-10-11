// SPY's daily closes (examples/spy.json) as ×100 log returns — what every example fits.
import { readFileSync } from 'node:fs';

const spy = JSON.parse(readFileSync(new URL('./spy.json', import.meta.url), 'utf8'));

export const lastDate = spy.last_date;
export const returns = spy.closes.slice(1).map((close, i) => Math.log(close / spy.closes[i]) * 100);
