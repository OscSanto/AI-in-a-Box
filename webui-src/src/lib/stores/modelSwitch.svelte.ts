/**
 * Tracks llama.cpp model switch state across the UI.
 * Set by DialogModelStore when set-model returns status:"starting".
 * Polled here so any component can read isSwitching/switchingTo.
 */

class ModelSwitchStore {
	isSwitching = $state(false);
	switchingTo = $state('');

	async startSwitch(model: string) {
		this.isSwitching = true;
		this.switchingTo = model;
		await this._poll(model);
		this.isSwitching = false;
		this.switchingTo = '';
	}

	private async _poll(model: string) {
		for (let i = 0; i < 90; i++) {
			await new Promise((r) => setTimeout(r, 2000));
			try {
				const res  = await fetch('/api/llamacpp/status');
				const data = await res.json();
				if (data.ready && data.model === model) return;
			} catch { /* server restarting */ }
		}
	}
}

export const modelSwitchStore = new ModelSwitchStore();
