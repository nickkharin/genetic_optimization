function sysCall_init()
    corout=coroutine.create(coroutineMain)
end

function sysCall_actuation()
    if coroutine.status(corout)~='dead' then
        local ok,errorMsg=coroutine.resume(corout)
        if errorMsg then
            error(debug.traceback(corout,errorMsg),2)
        end
    end
end

function coroutineMain()
    context=simZMQ.ctx_new()
    responder=simZMQ.socket(context,simZMQ.REP)
    local rc=simZMQ.bind(responder,'tcp://*:5555')
    if rc~=0 then error('failed bind') end

    while true do
        local rc,data=simZMQ.recv(responder,0)
        print('[responder] Received "' .. data .. '"')
        local response = processRequest(data)
        print('[responder] Sending "' .. response .. '"...')
        simZMQ.send(responder,response,0)
    end
end

function processRequest(request)
    local cmd, params = string.match(request, "^(%w+):(.*)")
    if cmd == "getObjectHandle" then
        local handle = sim.getObject(params)
        if handle == -1 then
            return "Error: Object not found"
        else
            return tostring(handle)
        end
    elseif cmd == "setObjectPosition" then
        local objectName, x, y, z = string.match(params, "(%d+),([%+-]?%d+%.?%d*),([%+-]?%d+%.?%d*),([%+-]?%d+%.?%d*)")
        if objectName and x and y and z then
            objectHandle, x, y, z = tonumber(objectName), tonumber(x), tonumber(y), tonumber(z)
            if objectHandle ~= -1 then
                sim.setObjectPosition(objectHandle, -1, {x, y, z})
                local newPos = {x, y, z}
                local checkPos = sim.getObjectPosition(objectHandle, -1)
                print("New position set to: ", newPos[1], newPos[2], newPos[3])
                print("Verified position: ", checkPos[1], checkPos[2], checkPos[3])
                return "Position set successfully"
            else
                return "Error: Object not found"
            end
        else
            return "Error: Incorrect parameters"
        end
    elseif cmd == "setJointTargetPosition" then
        local jointHandle, position = string.match(params, "(%d+),([%+-]?%d+%.?%d*)")
        jointHandle, position = tonumber(jointHandle), tonumber(position)
        if jointHandle ~= -1 then
            sim.setJointTargetPosition(jointHandle, position)
            local currentPosition = sim.getJointPosition(jointHandle)
            print("Target position set to: ", position)
            print("Current position after set: ", currentPosition)
            return "OK"
        else
            return "Error: Joint handle not found"
        end
    elseif cmd == "getVersion" then
        local version = sim.getInt32Parameter(sim.intparam_program_version)
        return tostring(version)
    else
        return "Unknown command"
    end
end

function sysCall_cleanup()
    simZMQ.close(responder)
    simZMQ.ctx_term(context)
end
