// Env variables
const begHour = 96.0
const endHour = 112.0
const interval = 2.0
const baseApiUrl = 'api'
// End Env variables

let token = getCookie('stork-auth');
checkTokenAndRedirectIfEmpty();

setInterval(function(){
    checkTokenAndRedirectIfEmpty();
}, 5000);

function checkTokenAndRedirectIfEmpty() {
    token = getCookie('stork-auth');
    if (window.location.pathname !== '/login' && token === '') {
        window.location.href = '/login';
    }
}

function login(event) {
    postLoginData(event.srcElement.username.value, event.srcElement.password.value, baseApiUrl);
}

function postLoginData(username, password, baseApiUrl) {
    let xhr = new XMLHttpRequest();
    xhr.open('POST', `${baseApiUrl}/login`, true);
    let formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    xhr.onload = (e) => {
        if (xhr.status === 200) {
            window.location.href = '/';
        } else {
            document.getElementById('login-form-username-password-error').classList.remove('hidden');
        }
    };
    xhr.onerror = (e) => alert('An error occurred!');
    xhr.send(formData);
}

const loginForm = document.getElementById('login-form');
if (loginForm) {
    loginForm.addEventListener('submit', (event) => {
        event.preventDefault();
        login(event);
    })
}

document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('.modal');
    var instances = M.Modal.init(elems, undefined);
});
document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('.collapsible');
    var instances = M.Collapsible.init(elems, undefined);
});

function getCookie(cookieName) {
    var name = cookieName + '=';
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');
    for(var i = 0; i <ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return '';
}

function isAnImage(file) {
    if (file && file.type) {
        return file.type.startsWith('image/jpeg') || file.type.startsWith('image/png') || file.type.startsWith('image/tiff');
    }

    return false;
};

function getFormData(files, data) {
    const formData = new FormData();
    for (const file of files) {
        if (isAnImage(file)) {
            formData.append('images', file, file.name);
        }
    }
    formData.append('data', JSON.stringify(data));

    return formData;
};

function postFormData(formData, baseApiUrl) {
    let xhr = new XMLHttpRequest();
    xhr.open('POST', `${baseApiUrl}/upload`, true);
    xhr.setRequestHeader('Authorization','Basic ' + token);

    const loaders = [...document.getElementsByClassName('loader')];
    loaders.map(x => x.classList.remove('hidden'));

    xhr.onload = function (e) {
        if (xhr.status === 200) {
            response = JSON.parse(e.target.response);
            showResultData(response);

            submitBtn.removeAttribute('disabled');
        } else {
            alert('An error occurred!');
            maternalAgeInput.removeAttribute('disabled');
            submitBtn.removeAttribute('disabled');
        }
        loaders.map(x => x.classList.add('hidden'));
    };
    xhr.onerror = () => {
      loaders.map(x => x.classList.add('hidden'));
    };

    xhr.send(formData);
};

function showResultData(data) {
    let commmonResultsHaveBeenSet = false;
    const commonResultElement = document.getElementById('common-results');
    Object.keys(data).forEach(resultName => {
        const resultElement = document.getElementById(`${resultName}-results`);
        const goodPercentage = (data[resultName].euploidProbablity * 100).toFixed(2);
        const poorPercentage = (100 - goodPercentage).toFixed(2);

        const bar = resultElement.getElementsByClassName('bar')[0];
        const goodText = resultElement.getElementsByClassName('good-text')[0];
        const poorText = resultElement.getElementsByClassName('poor-text')[0];
        bar.setAttribute('style', `width:${goodPercentage}%;`);
        goodText.innerHTML = `${goodPercentage}%`;
        poorText.innerHTML = `${poorPercentage}%`;
        if (data[resultName].euploidPrediction) {
            resultElement.getElementsByClassName('good-result-text')[0].classList.remove('hidden');
        } else {
            resultElement.getElementsByClassName('poor-result-text')[0].classList.remove('hidden');
        }

        if (!commmonResultsHaveBeenSet) {
            commonResultElement.getElementsByClassName('blastocystScore-text')[0].innerHTML = data[resultName].blastocystScore.toFixed(5);
            commonResultElement.getElementsByClassName('expansionScore-text')[0].innerHTML = data[resultName].expansionScore.toFixed(5);
            commonResultElement.getElementsByClassName('icmScore-text')[0].innerHTML = data[resultName].icmScore.toFixed(5);
            commonResultElement.getElementsByClassName('trophectodermScore-text')[0].innerHTML = data[resultName].trophectodermScore.toFixed(5);
            commmonResultsHaveBeenSet = true;
        }
    });
}

function clearResultData() {
    [...document.getElementsByClassName('bar')].forEach(e => e.setAttribute('style', 'width:0%'));
    ['good-result-text','poor-result-text'].forEach(className =>
        [...document.getElementsByClassName(className)].forEach(e => e.classList.add('hidden')));
    ['good-text','poor-text','blastocystScore-text','expansionScore-text','icmScore-text','trophectodermScore-text'].forEach(className =>
        [...document.getElementsByClassName(className)].forEach(e => e.innerHTML = '<span class="new badge" data-badge-caption="">N/A</span>'));
}

function filterClosestHours(objects) {
    //sort the list by the hour property  objects.sort((a, b) => a.hour - b.hour);
    let closestObjects = {};
    for (let obj of objects) {
      let hour = obj.hour;
      let closestHour = Array.from({length: minImagesRequired}, (_, i) => 96 + i*2).reduce((a, b) => Math.abs(b - hour) < Math.abs(a - hour) ? b : a);
      if (Math.abs(closestHour - hour) < 2) {
        if (!closestObjects[closestHour]) {
          closestObjects[closestHour] = obj;
        } else {
          if (Math.abs(closestObjects[closestHour].hour - closestHour) > Math.abs(hour - closestHour)) {
            closestObjects[closestHour] = obj;
          }
        }
      }
    }
    return Object.values(closestObjects);
  }

function removeImageCard(imageName) {
    const card = document.getElementById(`image-card-${imageName}`);
    imagesPlaceholder.removeChild(card);
    delete currentImages[imageName];

    imageRemovedUpdateUI();
    updateSelectedImages();
    updateSubmitBtn();
};

function imageRemovedUpdateUI() {
    if (Object.keys(currentImages).length === 0) {
        clearAllButton.classList.add('disabled');
    }
};

function clearAllImageCards() {
    Object.keys(currentImages).map(image => {
        removeImageCard(image);
    });
    clearResultData();
};

function createImageUIFromFile(file, imagesPlaceholder) {
    if (file === null || file === undefined || imagesPlaceholder === null || imagesPlaceholder === undefined) {
        return;
    }

    const imagePlaceholder = document.createElement('div');
    imagePlaceholder.classList.add('image-card');
    imagePlaceholder.id = `image-card-${file.name}`;

    imagePlaceholder.innerHTML = `
        <div class="card image-container">
            <div class="card-image">
                <img alt="${file.name}" width="330px" />
            </div>
            <div class="card-file-name">${file.name}</div>

            <div class="selected-image-mark hidden">
                <i class="material-icons">check_circle</i>
            </div>
            <div class="delete-image-button" onclick="removeImageCard('${file.name}')">
                <i class="material-icons">clear</i>
            </div>

        </div>`;
    imagesPlaceholder.appendChild(imagePlaceholder);

    const image = imagePlaceholder.getElementsByTagName('img')[0];
    const reader = new FileReader();
    reader.onloadend = function(event) {
        const arrayBuffer = reader.result;
        const blob = new Blob([arrayBuffer], {type: 'image/png'});
        image.src = URL.createObjectURL(blob);
    };

    reader.readAsArrayBuffer(file);
};

function submit() {
    const data = {
      maternalAge
    };

    submitBtn.setAttribute('disabled', 'disabled');
    maternalAgeInput.setAttribute('disabled', 'disabled');
    const filesToBeUploaded = selectedImages.map(x => currentImages[x]);
    postFormData(getFormData(filesToBeUploaded, data), baseApiUrl);
};

function updateSelectedImages() {
    if (currentImages && Object.keys(currentImages).length) {
        const objects = Object.keys(currentImages).map(filename => {
            let hour = null;
            let focus = null;
            const filenameParts = filename.split('.').slice(0,-1).join('.').split('_');
            
            if (filenameParts && filenameParts.length >= 5) {
                hour = filenameParts[2];
                focus = filenameParts[4];
            }

            return { filename, hour, focus };
        }).filter(x => x.focus == 0);

        selectedImages = filterClosestHours(objects).map(x => x.filename);
    }
    updateSelectedImagesUI();
}

function updateSelectedImagesUI() {
    Object.keys(currentImages).forEach(filename => {
        const element =  document.querySelector(`#image-card-${CSS.escape(filename)} .selected-image-mark`);
        if (selectedImages.includes(filename)) {
            element.classList.remove('hidden');
        } else {
            element.classList.add('hidden');
        }
    });


    selectedImagesElement.innerHTML = selectedImages.map(x => {
        return `<div>${x}</div>`;
    }).join('');
}

function updateSubmitBtn() {
    if (maternalAge && selectedImages.length >= minImagesRequired) {
        submitBtn.classList.remove('disabled');
    } else {
        submitBtn.classList.add('disabled');
    }
};

function updateClearAllButton() {
    if (currentImages &&  Object.keys(currentImages).length) {
        clearAllButton.classList.remove('disabled');
    } else {
        clearAllButton.classList.add('disabled');
    }
}

function handleFiles(files) {
    if (files && files.length) {
        for (const file of files) {
            if (currentImages[file.name]) continue;

            if (isAnImage(file)) {
                currentImages[file.name] = file;
                createImageUIFromFile(file, imagesPlaceholder);
            }
        }
        
        updateSelectedImages();
        updateClearAllButton();
        updateSubmitBtn();
    }
};

function dropHandler(event) {
    handleFiles(event.dataTransfer.files);
};

function preventDefaults (e) {
    e.preventDefault();
    e.stopPropagation();
}

clearResultData();
const form = document.getElementById('file-form');
const currentImages = {};
let selectedImages = [];
let maternalAge = null;
const minImagesRequired = ((endHour - begHour) / interval) + 1;
const selectedImagesElement = document.querySelector('#selected-images .collapsible-body');
const resultsElement = document.getElementById('results-placeholder');
const fileSelect = document.getElementById('file-select');
if (fileSelect) {
    fileSelect.value = '';
    fileSelect.addEventListener('change', function(event) {
        handleFiles(event.target.files);
    });
}

const maternalAgeInput = document.getElementById('maternal-age');
if (maternalAgeInput) {
    maternalAgeInput.addEventListener('change', function(event) {
        maternalAge = event.target.value
        ? parseFloat(event.target.value)
        : null;
    
        updateSubmitBtn();
    });
}

const submitBtn = document.getElementById('submit-btn');
if (submitBtn) {
    submitBtn.addEventListener('click', () => { submit(); } );
}

const clearAllButton = document.getElementById('clear-all-button');
if (clearAllButton) {
    clearAllButton.addEventListener('click', () => { clearAllImageCards(); } );
}

const imagesPlaceholder = document.getElementById('imageCards-placeholder');
if (imagesPlaceholder) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        imagesPlaceholder.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        imagesPlaceholder.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        imagesPlaceholder.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        imagesPlaceholder.classList.add('highlight');
    };

    function unhighlight(e) {
        imagesPlaceholder.classList.remove('highlight');
    };
}