/**
 * Stores the Wikipedia articles retrieved by the most recent pipeline run.
 * Populated by fetching /sources after each query completes.
 */

export interface ArticleSource {
	title: string;
	url: string;
	snippet: string;
}

let _articles = $state<ArticleSource[]>([]);

export function articles(): ArticleSource[] {
	return _articles;
}

export async function fetchSources(): Promise<void> {
	try {
		const res = await fetch('/sources');
		const data = await res.json();
		_articles = data.articles ?? [];
	} catch {
		// non-fatal — panel stays empty on error
	}
}

export function setArticles(sources: ArticleSource[]): void {
	_articles = sources;
}

export function clearSources(): void {
	_articles = [];
}
