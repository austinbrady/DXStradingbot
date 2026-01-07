/**
 * DXS HFT Bot - Simple coordinator
 */

console.log('🚀 DXS Bot Started');

let botState = {
    state: 'STOPPED',
    trades_executed: 0,
    daily_pnl: 0,
    active_positions: 0,
    startTime: null
};

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    switch (request.action) {
        case 'BOT_START':
            botState.state = 'RUNNING';
            botState.startTime = Date.now();
            botState.trades_executed = 0;
            botState.daily_pnl = 0;
            console.log('✅ Bot Started');
            sendResponse({ status: 'Running' });
            break;

        case 'BOT_STOP':
            botState.state = 'STOPPED';
            console.log('🛑 Bot Stopped');
            sendResponse({ status: 'Stopped' });
            break;

        case 'BOT_PAUSE':
            botState.state = 'PAUSED';
            sendResponse({ status: 'Paused' });
            break;

        case 'BOT_RESUME':
            botState.state = 'RUNNING';
            sendResponse({ status: 'Running' });
            break;

        case 'GET_STATE':
            sendResponse({ state: botState });
            break;

        case 'TRADE_EXECUTED':
            botState.trades_executed++;
            botState.active_positions++;
            sendResponse({ ok: true });
            break;

        case 'TRADE_CLOSED':
            botState.active_positions--;
            botState.daily_pnl += request.pnl;
            sendResponse({ ok: true });
            break;

        default:
            sendResponse({ error: 'Unknown' });
    }
    return true;
});

setInterval(() => {
    if (botState.state === 'RUNNING' && botState.startTime) {
        chrome.tabs.query({ url: 'https://dxs.app/*' }, (tabs) => {
            if (tabs.length > 0) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    action: 'EXECUTE_TRADING_CYCLE',
                    state: botState
                }).catch(() => {});
            }
        });
    }
}, 500); // Run every 500ms