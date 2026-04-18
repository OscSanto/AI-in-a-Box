/**
 * Stores the Wikipedia articles retrieved by the most recent pipeline run.
 * Populated via the inline SSE sources chunk emitted before the first text token.
 */

export interface ArticleSource {
	title: string;
	url: string;
	snippet: string;
	section?: string;
	context?: string;
	rerank_score?: number;
	faiss_score?: number;
	zim_name?: string;
}

let _articles = $state<ArticleSource[]>([]);

export function articles(): ArticleSource[] {
	return _articles;
}

export function setArticles(sources: ArticleSource[]): void {
	_articles = sources;
}

export function clearSources(): void {
	_articles = [];
}
