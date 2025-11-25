const folderList = document.getElementById('folder-list');
const fileList = document.getElementById('file-list');
const currentFolderTitle = document.getElementById('current-folder-title');
const audioPlayer = document.getElementById('audio-player');
const trackName = document.getElementById('track-name');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const autoplayCheck = document.getElementById('autoplay-check');
const playlistStats = document.getElementById('playlist-stats');
const playlistProgress = document.getElementById('playlist-progress');
const playlistDuration = document.getElementById('playlist-duration');
const currentTimeDisplay = document.getElementById('current-time');
const totalTimeDisplay = document.getElementById('total-time');
const downloadBtn = document.getElementById('download-btn');
const folderSortSelect = document.getElementById('folder-sort');
const fileSortSelect = document.getElementById('file-sort');

let currentFolder = null;
let currentFiles = [];
let currentFileIndex = -1;

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatDate(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
}

// Load folders on startup
async function loadFolders() {
    try {
        const sort = folderSortSelect.value;
        const response = await fetch(`/api/folders?sort=${sort}`);
        const folders = await response.json();
        
        folderList.innerHTML = '';
        folders.forEach(folder => {
            const li = document.createElement('li');
            li.innerHTML = `
                <div class="folder-name">${folder.name}</div>
                <div class="folder-date">${formatDate(folder.created)}</div>
            `;
            li.onclick = () => selectFolder(folder.name);
            folderList.appendChild(li);
        });
    } catch (error) {
        console.error('Error loading folders:', error);
    }
}

async function selectFolder(folder) {
    currentFolder = folder;
    currentFolderTitle.textContent = folder;
    playlistStats.classList.remove('stats-hidden');
    downloadBtn.classList.remove('hidden');
    
    // Update active state in sidebar
    Array.from(folderList.children).forEach(li => {
        const nameDiv = li.querySelector('.folder-name');
        li.classList.toggle('active', nameDiv && nameDiv.textContent === folder);
    });

    await loadFiles();
}

async function loadFiles() {
    if (!currentFolder) return;
    
    try {
        const sort = fileSortSelect.value;
        const response = await fetch(`/api/files/${currentFolder}?sort=${sort}`);
        const files = await response.json();
        currentFiles = files;
        
        // Calculate total duration
        const totalSeconds = currentFiles.reduce((acc, file) => acc + file.duration, 0);
        playlistDuration.textContent = `Total: ${formatTime(totalSeconds)}`;
        
        renderFileList();
        updatePlaylistProgress();
        
        // If we just switched folders, maybe play first track? 
        // Or if we just re-sorted, maybe keep playing current track if possible?
        // For now, let's just reset if it's a new folder load or explicit sort.
        // Actually, if we re-sort, the index changes. 
        // If audio is playing, we should try to find the current track in the new list and update index.
        
        if (audioPlayer.src && !audioPlayer.paused) {
             // Find current track name
             const currentTrackName = decodeURIComponent(audioPlayer.src.split('/').pop());
             const newIndex = currentFiles.findIndex(f => f.name === currentTrackName);
             if (newIndex !== -1) {
                 currentFileIndex = newIndex;
                 // Don't interrupt playback, just update UI
             }
        } else if (currentFiles.length > 0) {
            // If not playing, select first but don't auto-play unless it's a fresh folder selection?
            // The user didn't specify behavior. Let's just select first.
            // playTrack(0); // This auto-plays. Maybe just set index to 0 and wait.
        } else {
            trackName.textContent = "No audio files found";
            audioPlayer.src = "";
        }
    } catch (error) {
        console.error('Error loading files:', error);
    }
}

function renderFileList() {
    fileList.innerHTML = '';
    currentFiles.forEach((file, index) => {
        const li = document.createElement('li');
        li.innerHTML = `
            <div class="file-info">
                <span class="file-name">${file.name}</span>
                <span class="file-date">${formatDate(file.created)}</span>
            </div>
            <span class="file-duration">${formatTime(file.duration)}</span>
        `;
        li.onclick = () => playTrack(index);
        if (index === currentFileIndex) {
            li.classList.add('playing');
        }
        fileList.appendChild(li);
    });
}

function updatePlaylistProgress() {
    playlistProgress.textContent = `Track ${currentFileIndex + 1} of ${currentFiles.length}`;
}

function playTrack(index) {
    if (index < 0 || index >= currentFiles.length) return;

    currentFileIndex = index;
    const file = currentFiles[index];
    trackName.textContent = file.name;
    
    // Highlight in list
    renderFileList();
    updatePlaylistProgress();
    
    // Set audio source
    audioPlayer.src = `/audio/${currentFolder}/${file.name}`;
    audioPlayer.play();
}

// Download handler
downloadBtn.onclick = () => {
    if (!currentFolder) return;
    
    const sort = fileSortSelect.value;
    // Trigger download
    window.location.href = `/api/convert/${currentFolder}?sort=${sort}`;
};

// Audio events
audioPlayer.addEventListener('timeupdate', () => {
    currentTimeDisplay.textContent = formatTime(audioPlayer.currentTime);
    totalTimeDisplay.textContent = formatTime(audioPlayer.duration || 0);
});

audioPlayer.addEventListener('ended', () => {
    if (autoplayCheck.checked) {
        if (currentFileIndex < currentFiles.length - 1) {
            playTrack(currentFileIndex + 1);
        }
    }
});

// Controls
prevBtn.onclick = () => playTrack(currentFileIndex - 1);
nextBtn.onclick = () => playTrack(currentFileIndex + 1);

// Sort listeners
folderSortSelect.onchange = loadFolders;
fileSortSelect.onchange = loadFiles;

loadFolders();
