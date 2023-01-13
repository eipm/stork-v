// Env variables
const begHour = 96.0
const endHour = 112.0
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
    Object.keys(data).forEach(resultName => {
        const resultElement = document.getElementById(`${resultName}-results`);
        resultElement.getElementsByClassName('blastocystScore-text')[0].innerHTML = data[resultName].blastocystScore;
        resultElement.getElementsByClassName('euploidPrediction-text')[0].innerHTML = data[resultName].euploidPrediction;
        resultElement.getElementsByClassName('euploidProbablity-text')[0].innerHTML = data[resultName].euploidProbablity;
        resultElement.getElementsByClassName('expansionScore-text')[0].innerHTML = data[resultName].expansionScore;
        resultElement.getElementsByClassName('icmScore-text')[0].innerHTML = data[resultName].icmScore;
        resultElement.getElementsByClassName('trophectodermScore-text')[0].innerHTML = data[resultName].trophectodermScore;
    });

    resultsElement.classList.remove('hidden');
}

function removeImageCard(imageName) {
    const card = document.getElementById(`image-card-${imageName}`);
    imagesPlaceholder.removeChild(card);
    delete currentImages[imageName];
    imageRemovedUpdateUI();
};

function imageRemovedUpdateUI() {
    if (Object.keys(currentImages).length === 0) {
        clearAllButton.classList.add('disabled');
    }
    
    updateSubmitBtn();
};

function clearAllImageCards() {
    Object.keys(currentImages).map(image => {
        removeImageCard(image);
    });
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
    const filesToBeUploaded = Object.keys(currentImages).map(x => currentImages[x]);
    postFormData(getFormData(filesToBeUploaded, data), baseApiUrl);
};

function updateSubmitBtn() {
    if (maternalAge && currentImages && 
        Object.keys(currentImages).length > minImagesRequired) {
            submitBtn.classList.remove('disabled');
    } else {
        submitBtn.classList.add('disabled');
    }
};

function handleFiles(files) {
    if (files && files.length) {
        for (const file of files) {
            if (currentImages[file.name]) continue;

            if (isAnImage(file)) {
                currentImages[file.name] = file;
                clearAllButton.classList.remove('disabled');
                createImageUIFromFile(file, imagesPlaceholder);
            }
        }
        
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

const form = document.getElementById('file-form');
const currentImages = {};
let maternalAge = null;
const minImagesRequired = 2 * (endHour - begHour);
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