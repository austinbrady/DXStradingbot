// Update popup with current bot state
chrome.runtime.sendMessage({ action: 'GET_STATE' }, (response) => {
    console.log('Bot state:', response);
});
```

### **6. Create Folder Structure**

Your extension folder should look like this:
```
~/Desktop/Dev/Apps_Local/DXS Trading Bot/
├── manifest.json
├── content.js
├── background.js
├── popup.html
├── popup.js
└── images/
    ├── icon-16.png
    ├── icon-48.png
    └── icon-128.png