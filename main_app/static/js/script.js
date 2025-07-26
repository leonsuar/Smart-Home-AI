// main_app/static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    const commandInput = document.getElementById('command-input');
    const sendCommandButton = document.getElementById('send-command-button');
    const logDisplay = document.getElementById('log-display');
    const systemInfoSection = document.getElementById('system-info-section');
    const discoveredEntitiesList = document.getElementById('discovered-entities-list');
    const tasmotaMapList = document.getElementById('tasmota-map-list');
    const saveConfirmationContainer = document.getElementById('save-confirmation-container');
    const confirmSaveYesButton = document.getElementById('confirm-save-yes');
    const confirmSaveNoButton = document.getElementById('confirm-save-no');

    // Elementos del cuadro de mensaje genérico (sigue siendo un modal)
    const messageBox = document.getElementById('messageBox');
    const messageText = document.getElementById('messageText');
    const messageBoxCloseButton = document.getElementById('messageBoxCloseButton');

    let isSavingConfirmed = false; // Bandera para evitar múltiples envíos de confirmación

    // Función para mostrar mensajes en un cuadro de diálogo personalizado
    function showMessageBox(message) {
        messageText.textContent = message;
        messageBox.classList.remove('hidden');
    }

    // Event listener para cerrar el cuadro de mensaje
    messageBoxCloseButton.addEventListener('click', () => {
        messageBox.classList.add('hidden');
    });

    // Función para añadir mensajes al log de la IA en la interfaz
    function addLogEntryToUI(message) {
        const div = document.createElement('div');
        div.className = `log-entry ${message.tipo}`; // Usa la clase CSS basada en el tipo
        div.innerHTML = `<strong>[${message.tiempo}]</strong> ${message.mensaje}`;
        logDisplay.appendChild(div);
        logDisplay.scrollTop = logDisplay.scrollHeight; // Auto-scroll al final
    }

    // Función para alternar la visibilidad de las secciones (para los toggles)
    window.toggleSection = function(contentId, iconId) { // Hacer global para onclick en HTML
        const content = document.getElementById(contentId);
        const icon = document.getElementById(iconId);
        if (content.classList.contains('expanded')) {
            content.classList.remove('expanded');
            icon.classList.remove('rotated');
        } else {
            content.classList.add('expanded');
            icon.classList.add('rotated');
        }
    };

    async function fetchLogAndState() {
        try {
            const response = await fetch('/obtener_log');
            const data = await response.json();

            // Actualizar Log
            // Limpiar logDisplay antes de añadir nuevos logs para evitar duplicados
            logDisplay.innerHTML = ''; 
            data.log.forEach(entry => addLogEntryToUI(entry));
            logDisplay.scrollTop = logDisplay.scrollHeight; // Auto-scroll al final

            // Actualizar Información del Sistema
            systemInfoSection.innerHTML = '';
            data.estado_red.filter(entry => entry.tipo && entry.tipo.startsWith("Sistema")).forEach(entry => {
                const p = document.createElement('p');
                p.className = 'text-sm text-gray-300';
                p.textContent = `${entry.tipo}: ${entry.valor}`; // Solo mostrar tipo y valor
                systemInfoSection.appendChild(p);
            });

            // Actualizar Dispositivos Descubiertos por HA
            discoveredEntitiesList.innerHTML = '';
            if (data.discovered_entities && Object.keys(data.discovered_entities).length > 0) {
                for (const entityId in data.discovered_entities) {
                    const info = data.discovered_entities[entityId];
                    const li = document.createElement('li');
                    li.className = 'device-list-item';
                    li.innerHTML = `<strong>${info.name || entityId}</strong> (${entityId})<br>
                                    Dominio: ${info.domain || 'N/A'} | Cmd Tópico: ${info.command_topic || 'N/A'}`;
                    discoveredEntitiesList.appendChild(li);
                }
            } else {
                const li = document.createElement('li');
                li.className = 'device-list-item';
                li.textContent = 'No se han descubierto dispositivos MQTT.';
                discoveredEntitiesList.appendChild(li);
            }

            // Actualizar Mapeo de Comandos Tasmota
            tasmotaMapList.innerHTML = '';
            if (data.tasmota_map && Object.keys(data.tasmota_map).length > 0) {
                for (const entityId in data.tasmota_map) {
                    const info = data.tasmota_map[entityId];
                    const li = document.createElement('li');
                    li.className = 'device-list-item';
                    li.innerHTML = `<strong>${entityId}</strong><br>
                                    Prefijo Comando: ${info.command_topic_prefix || 'N/A'} | Sufijo Power: ${info.power_topic_suffix || 'N/A'}`;
                    tasmotaMapList.appendChild(li);
                }
            } else {
                const li = document.createElement('li');
                li.className = 'device-list-item';
                li.textContent = 'No hay mapeos de comandos Tasmota específicos.';
                tasmotaMapList.appendChild(li);
            }

        } catch (error) {
            console.error('Error al obtener log y estado:', error);
            // No mostrar showMessageBox aquí para evitar que aparezca constantemente en errores de red
            // addLogEntryToUI({ tiempo: new Date().toISOString().slice(0, 19).replace('T', ' '), tipo: 'error', fuente: 'System', mensaje: `Error al conectar con el servidor: ${error.message}` });
        }
    }

    async function sendCommand() {
        const command = commandInput.value.trim();
        if (!command) {
            showMessageBox('Por favor, introduce un comando.');
            return;
        }

        // Añadir el comando del usuario al log de la UI inmediatamente
        addLogEntryToUI({ tiempo: new Date().toISOString().slice(0, 19).replace('T', ' '), tipo: 'comando', fuente: 'User', mensaje: command });
        commandInput.value = ''; // Limpiar el input

        try {
            const response = await fetch('/enviar_comando', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ comando: command })
            });
            const data = await response.json();
            
            // La actualización del log la maneja fetchLogAndState() en el intervalo.
            // Solo necesitamos manejar la confirmación de guardado.

            if (data.should_offer_to_save) {
                showSaveConfirmation();
            } else {
                // Si no se ofrece guardar, asegurar que el contenedor de confirmación esté oculto
                saveConfirmationContainer.classList.add('hidden');
            }

        } catch (error) {
            console.error('Error al enviar comando:', error);
            showMessageBox('Error de comunicación con el servidor. Revisa la consola.');
            addLogEntryToUI({ tiempo: new Date().toISOString().slice(0, 19).replace('T', ' '), tipo: 'error', fuente: 'System', mensaje: `Error al enviar comando: ${error.message}` });
        }
    }

    function showSaveConfirmation() {
        isSavingConfirmed = false; // Resetear la bandera para la nueva interacción
        saveConfirmationContainer.classList.remove('hidden'); // Mostrar el contenedor de confirmación
    }

    async function sendSaveChoice(choice) {
        if (isSavingConfirmed) return; // Evitar múltiples envíos
        isSavingConfirmed = true;

        saveConfirmationContainer.classList.add('hidden'); // Ocultar el contenedor de confirmación después de la elección

        try {
            const response = await fetch('/confirm_save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ choice: choice })
            });
            const data = await response.json();
            showMessageBox(data.message); // Mostrar el mensaje de éxito/error en el modal general
        } catch (error) {
            console.error('Error al enviar elección de guardado:', error);
            showMessageBox('Error al enviar elección de guardado: ' + error.message);
        }
    }

    // Event Listeners
    sendCommandButton.addEventListener('click', sendCommand);
    commandInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendCommand();
        }
    });

    confirmSaveYesButton.addEventListener('click', () => sendSaveChoice('yes'));
    confirmSaveNoButton.addEventListener('click', () => sendSaveChoice('no'));

    // Cargar log y estado al iniciar y cada pocos segundos
    fetchLogAndState();
    setInterval(fetchLogAndState, 2000); // Actualizar cada 2 segundos
});
